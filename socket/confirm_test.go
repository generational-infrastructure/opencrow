package socket

import (
	"context"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/pinpox/opencrow/backend"
)

// noopHandler is a no-op MessageHandler used by confirm tests that don't
// exercise the inbound message path.
func noopHandler(_ context.Context, _ backend.Message) {}

// readConfirmEvent reads events from sc until an evConfirm arrives,
// skipping any unrelated kinds (status, models, etc.).
func readConfirmEvent(t *testing.T, sc *connScanner, conn net.Conn) event {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for confirm event")
		}
		ev := sc.readEvent(t, conn)
		if ev.Kind == evConfirm {
			return ev
		}
	}
}

func TestRequestConfirm_AllowRoundTrip(t *testing.T) {
	t.Parallel()

	b, sockPath, cancel := startBackend(t, noopHandler, &stubModelService{})
	defer cancel()

	conn := dialSynced(t, sockPath)
	defer conn.Close()

	sc := newConnScanner(conn)

	type result struct {
		ok  bool
		err error
	}
	resCh := make(chan result, 1)
	go func() {
		ok, err := b.RequestConfirm(t.Context(), "req-1", "Run shell command?", "echo hi", 30000)
		resCh <- result{ok: ok, err: err}
	}()

	ev := readConfirmEvent(t, sc, conn)
	if ev.ConfirmID != "req-1" {
		t.Errorf("ConfirmID = %q, want req-1", ev.ConfirmID)
	}
	if ev.ConfirmTitle != "Run shell command?" {
		t.Errorf("ConfirmTitle = %q", ev.ConfirmTitle)
	}
	if ev.ConfirmBody != "echo hi" {
		t.Errorf("ConfirmBody = %q", ev.ConfirmBody)
	}

	sendCommand(t, conn, command{Cmd: cmdConfirmResponse, ID: "req-1", Confirmed: true})

	select {
	case r := <-resCh:
		if r.err != nil {
			t.Fatalf("RequestConfirm err: %v", r.err)
		}
		if !r.ok {
			t.Fatal("RequestConfirm returned false, want true")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("RequestConfirm did not return after confirm-response")
	}
}

func TestRequestConfirm_DenyRoundTrip(t *testing.T) {
	t.Parallel()

	b, sockPath, cancel := startBackend(t, noopHandler, &stubModelService{})
	defer cancel()

	conn := dialSynced(t, sockPath)
	defer conn.Close()

	sc := newConnScanner(conn)

	resCh := make(chan bool, 1)
	go func() {
		ok, _ := b.RequestConfirm(t.Context(), "req-deny", "t", "rm -rf /", 30000)
		resCh <- ok
	}()

	ev := readConfirmEvent(t, sc, conn)
	if ev.ConfirmID != "req-deny" {
		t.Fatalf("ConfirmID = %q", ev.ConfirmID)
	}

	sendCommand(t, conn, command{Cmd: cmdConfirmResponse, ID: "req-deny", Confirmed: false})

	select {
	case ok := <-resCh:
		if ok {
			t.Fatal("RequestConfirm returned true, want false")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("RequestConfirm did not return")
	}
}

func TestRequestConfirm_EmptyIDRejected(t *testing.T) {
	t.Parallel()

	b, _, cancel := startBackend(t, noopHandler, &stubModelService{})
	defer cancel()

	ok, err := b.RequestConfirm(t.Context(), "", "t", "m", 0)
	if err == nil {
		t.Fatal("expected error for empty id")
	}
	if ok {
		t.Fatal("ok = true for invalid call")
	}
}

func TestRequestConfirm_ContextCancel(t *testing.T) {
	t.Parallel()

	b, sockPath, cancel := startBackend(t, noopHandler, &stubModelService{})
	defer cancel()

	conn := dialSynced(t, sockPath)
	defer conn.Close()

	sc := newConnScanner(conn)

	ctx, cancelReq := context.WithCancel(t.Context())

	resCh := make(chan error, 1)
	go func() {
		_, err := b.RequestConfirm(ctx, "req-ctx", "t", "m", 0)
		resCh <- err
	}()

	// Wait for the event so we know the call is parked on the pending channel.
	_ = readConfirmEvent(t, sc, conn)

	cancelReq()

	select {
	case err := <-resCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("RequestConfirm did not return after ctx cancel")
	}

	// Pending entry must be cleaned up so a late stale response is a no-op.
	b.pendingMu.Lock()
	_, stillPending := b.pending["req-ctx"]
	b.pendingMu.Unlock()
	if stillPending {
		t.Fatal("pending entry not cleaned after ctx cancel")
	}

	// Stale response is dropped (no panic, no deliver).
	sendCommand(t, conn, command{Cmd: cmdConfirmResponse, ID: "req-ctx", Confirmed: true})
	time.Sleep(50 * time.Millisecond)
}

func TestRequestConfirm_UnknownIDIgnored(t *testing.T) {
	t.Parallel()

	_, sockPath, cancel := startBackend(t, noopHandler, &stubModelService{})
	defer cancel()

	conn := dialSynced(t, sockPath)
	defer conn.Close()

	// No outstanding RequestConfirm; server must not crash on stale resp.
	sendCommand(t, conn, command{Cmd: cmdConfirmResponse, ID: "ghost", Confirmed: true})

	// Round-trip a replay to confirm the connection (and server) is still alive.
	if _, err := conn.Write([]byte(`{"cmd":"replay"}` + "\n")); err != nil {
		t.Fatalf("post-stale write: %v", err)
	}
	sc := newConnScanner(conn)
	_ = sc.readEvent(t, conn) // status reply
}

func TestRequestConfirm_FirstResponseWins(t *testing.T) {
	t.Parallel()

	b, sockPath, cancel := startBackend(t, noopHandler, &stubModelService{})
	defer cancel()

	c1 := dialSynced(t, sockPath)
	defer c1.Close()
	c2 := dialSynced(t, sockPath)
	defer c2.Close()

	sc1 := newConnScanner(c1)
	sc2 := newConnScanner(c2)

	resCh := make(chan bool, 1)
	go func() {
		ok, _ := b.RequestConfirm(t.Context(), "race", "t", "m", 0)
		resCh <- ok
	}()

	// Both clients must see the event.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); _ = readConfirmEvent(t, sc1, c1) }()
	go func() { defer wg.Done(); _ = readConfirmEvent(t, sc2, c2) }()
	wg.Wait()

	// First client answers allow.
	sendCommand(t, c1, command{Cmd: cmdConfirmResponse, ID: "race", Confirmed: true})

	select {
	case ok := <-resCh:
		if !ok {
			t.Fatal("first response should have won (true)")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("RequestConfirm did not return")
	}

	// Late deny from the other client must be ignored (no double-deliver).
	sendCommand(t, c2, command{Cmd: cmdConfirmResponse, ID: "race", Confirmed: false})
	time.Sleep(50 * time.Millisecond)
}
