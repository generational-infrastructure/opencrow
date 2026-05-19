package main

import (
	"context"
	"testing"
	"time"
)

// TestWaitForResult_StreamsDeltasInRealTime pins down the streaming
// contract on the rpc layer: every message_update event carrying a
// text_delta must invoke onDelta before the next event is consumed
// from the channel. A regression that buffers deltas inside the
// rpc/worker pipeline (e.g. collecting them and flushing only on
// agent_end) would starve the per-event receive below and time out.
//
// Channel buffer is intentionally 1 so a stuck reader cannot mask the
// bug by absorbing several events ahead of any delta callback firing.
func TestWaitForResult_StreamsDeltasInRealTime(t *testing.T) {
	t.Parallel()

	ch := make(chan rpcParsed, 1)
	p := &PiProcess{events: ch, done: make(chan struct{})}

	deltas := make(chan string, 16)
	onDelta := func(s string) { deltas <- s }

	type waitResult struct {
		reply string
		err   error
	}
	doneCh := make(chan waitResult, 1)
	go func() {
		reply, err := p.waitForResult(context.Background(), onDelta)
		doneCh <- waitResult{reply, err}
	}()

	pieces := []string{"Hello", ", ", "world", "!"}
	for _, s := range pieces {
		ch <- rpcParsed{event: rpcEvent{
			Type: rpcTypeMessageUpdate,
			AssistantMessageEvent: &assistantMessageEvent{
				Type:  "text_delta",
				Delta: s,
			},
		}}

		select {
		case got := <-deltas:
			if got != s {
				t.Fatalf("delta = %q, want %q", got, s)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timeout waiting for delta %q — streaming pipeline buffered the event", s)
		}
	}

	// Terminate the turn so waitForResult returns.
	ch <- rpcParsed{event: agentEnd("end_turn", "", "Hello, world!")}

	select {
	case r := <-doneCh:
		if r.err != nil {
			t.Fatalf("waitForResult: %v", r.err)
		}
		if r.reply != "Hello, world!" {
			t.Fatalf("reply = %q, want %q", r.reply, "Hello, world!")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("waitForResult never returned after agent_end")
	}

	// Ensure no stray deltas leaked after agent_end.
	select {
	case extra := <-deltas:
		t.Fatalf("unexpected delta after agent_end: %q", extra)
	default:
	}
}

// TestWaitForResult_NoDeltaWhenNilCallback guards against the trivial
// regression where the message_update branch unconditionally
// dereferences onDelta even when nil.
func TestWaitForResult_NoDeltaWhenNilCallback(t *testing.T) {
	t.Parallel()

	ch := make(chan rpcParsed, 4)
	p := &PiProcess{events: ch, done: make(chan struct{})}

	ch <- rpcParsed{event: rpcEvent{
		Type: rpcTypeMessageUpdate,
		AssistantMessageEvent: &assistantMessageEvent{
			Type:  "text_delta",
			Delta: "x",
		},
	}}
	ch <- rpcParsed{event: agentEnd("end_turn", "", "ok")}

	reply, err := p.waitForResult(context.Background(), nil)
	if err != nil {
		t.Fatalf("waitForResult: %v", err)
	}

	if reply != "ok" {
		t.Fatalf("reply = %q, want ok", reply)
	}
}

// streamingBackend is a backend.Streamer test double that captures
// every SendDelta call so worker-level streaming tests can assert
// each fragment arrived in order without buffering.
type streamingBackend struct {
	stubBackend
	deltas chan string
}

func (s *streamingBackend) SendDelta(_ context.Context, _ string, _ string, delta string) {
	s.deltas <- delta
}

// TestWorker_StreamsDeltasToBackend pins down the end-to-end streaming
// contract: when pi emits text_delta message_update events during a
// prompt, the worker's onDelta hook MUST forward each fragment to the
// backend's SendDelta before the agent_end. Any regression that
// drops, buffers, or collapses deltas into a single post-completion
// flush (the visible symptom: chat bubbles appearing in one chunk
// instead of streaming) will starve the per-delta receive below.
func TestWorker_StreamsDeltasToBackend(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	be := &streamingBackend{deltas: make(chan string, 16)}
	w.SetBackend(be)

	// Mirror processPrompt's wiring: when the backend implements
	// backend.Streamer, onDelta forwards each delta to SendDelta.
	// Driving sendWithRetry directly (rather than the full
	// processPrompt → buildPrompt → app.sendReply chain) keeps the
	// test focused on the streaming pipeline.
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	onDelta := func(delta string) {
		be.SendDelta(ctx, "room", "stream-test", delta)
	}

	type sendResult struct {
		reply string
		err   error
	}
	done := make(chan sendResult, 1)
	go func() {
		_, reply, err := w.sendWithRetry(ctx, "stream-deltas please", onDelta)
		done <- sendResult{reply, err}
	}()

	// Drain four deltas. Each must arrive within the timeout
	// independently — buffering would surface as a timeout on the
	// first receive while the fake-pi script is still emitting the
	// rest.
	want := []string{"Hello", ", ", "world", "!"}
	for _, piece := range want {
		select {
		case got := <-be.deltas:
			if got != piece {
				t.Fatalf("delta = %q, want %q", got, piece)
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("timeout waiting for delta %q — streaming did not reach backend", piece)
		}
	}

	select {
	case r := <-done:
		if r.err != nil {
			t.Fatalf("sendWithRetry: %v", r.err)
		}
		if r.reply != "Hello, world!" {
			t.Fatalf("reply = %q, want %q", r.reply, "Hello, world!")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("sendWithRetry never returned")
	}
}
