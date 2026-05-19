package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"

	"github.com/pinpox/opencrow/backend"
)

type stubBackend struct{}

func (stubBackend) SetTyping(context.Context, string, bool)                    {}
func (stubBackend) SendMessage(context.Context, string, string, string) string { return "" }
func (stubBackend) MarkdownFlavor() backend.MarkdownFlavor                     { return backend.MarkdownNone }

// newFakePiWorker builds a Worker wired to the bash fake-pi stub.
// Cleanup stops the spawned process.
func newFakePiWorker(t *testing.T) *Worker {
	t.Helper()

	dir := t.TempDir()

	script, err := filepath.Abs("testdata/fake-pi")
	if err != nil {
		t.Fatal(err)
	}

	w := NewWorker(newTestInbox(t.Context(), t), PiConfig{
		BinaryPath: "bash",
		BinaryArgs: []string{script},
		SessionDir: dir,
		WorkingDir: dir,
	}, "", "")
	w.SetBackend(stubBackend{})
	w.SetRoomID("room")

	t.Cleanup(w.stopPi)

	return w
}

// Regression for the "No active session to compact" bug seen on eve:
// pi was spawned with exec.CommandContext bound to the per-item ctx,
// which is cancelled the moment processItem returns. Go's CommandContext
// then SIGKILLs pi, so by the time the next message (or !compact)
// arrives the worker reports IsActive() == false. Journald showed
// "pi: process exited" immediately after every "agent finished".
//
// The pi process lifetime is owned by the worker (idle reaper / Restart
// / Run shutdown via stopPi), not by an individual prompt's context.
func TestWorker_PiSurvivesItemContext(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	// Mirror processItem: a per-item ctx that is cancelled as soon as
	// the prompt completes.
	itemCtx, cancel := context.WithCancel(context.Background())

	pi, reply, err := w.sendWithRetry(itemCtx, "hello", nil)
	if err != nil || reply != "ok" {
		t.Fatalf("sendWithRetry = (%q, %v), want (ok, nil)", reply, err)
	}

	cancel() // processItem's defer cancel()

	// CommandContext kills via a watcher goroutine after <-ctx.Done(),
	// and IsAlive() reads p.done which a second goroutine closes after
	// cmd.Wait(). Neither has run immediately after cancel() returns,
	// so wait briefly for the bug to manifest if it's going to.
	select {
	case <-pi.done:
		t.Fatal("pi process died after item ctx cancel; it must outlive individual prompts")
	case <-time.After(500 * time.Millisecond):
	}

	// A follow-up prompt must reuse the existing process, not respawn.
	// On eve the dead process triggered sendWithRetry's "pi exited,
	// starting fresh process" path on every message.
	pi2, _, err := w.sendWithRetry(t.Context(), "again", nil)
	if err != nil {
		t.Fatalf("second sendWithRetry: %v", err)
	}

	if pi2 != pi {
		t.Fatal("second prompt spawned a new pi process; expected reuse")
	}

	// And the user-facing symptom: !compact must find a live session.
	cr, err := pi2.Compact(t.Context())
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	if cr.Summary != "s" || cr.TokensBefore != 1 {
		t.Fatalf("compact result = %+v", cr)
	}
}

// Regression for slow container restarts after 0f1495a: that commit
// moved pi off the per-item context (correct) onto context.Background,
// removing the only thing that force-killed the process tree on
// shutdown. Kill() sent SIGINT/SIGKILL to the pi leader only, so any
// tool subprocess pi had forked (bash, curl, nix build…) was orphaned
// and reparented to init. systemd then waited TimeoutStopSec (90s) for
// the cgroup to empty before SIGKILLing it.
//
// fake-pi's "spawn-child" prompt forks a `sleep 300` grandchild and
// writes its PID; the test asserts stopPi reaps it.
func TestWorker_StopPiKillsProcessTree(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	_, _, err := w.sendWithRetry(t.Context(), "spawn-child", nil)
	if err != nil {
		t.Fatalf("sendWithRetry: %v", err)
	}

	pidBytes, err := os.ReadFile(filepath.Join(w.piCfg.SessionDir, "child.pid"))
	if err != nil {
		t.Fatalf("reading child.pid (fake-pi should have written it): %v", err)
	}

	pid, err := strconv.Atoi(string(pidBytes))
	if err != nil || pid <= 0 {
		t.Fatalf("child.pid = %q", pidBytes)
	}

	// Sanity: grandchild is alive before shutdown.
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("grandchild %d not running before stopPi: %v", pid, err)
	}

	w.stopPi()

	// Kill() escalates to SIGKILL after 5s; allow a little slack for
	// the kernel to reap. Poll instead of sleeping the full window so
	// the happy path stays fast.
	deadline := time.Now().Add(6 * time.Second)

	for {
		err := syscall.Kill(pid, 0)
		if errors.Is(err, syscall.ESRCH) {
			return
		}

		if time.Now().After(deadline) {
			// Don't leak the sleep into the test runner.
			_ = syscall.Kill(pid, syscall.SIGKILL)
			t.Fatalf("grandchild %d still alive after stopPi; "+
				"pi process tree leaks on shutdown (kill(pid,0) = %v)", pid, err)
		}

		time.Sleep(50 * time.Millisecond)
	}
}

// TestWorker_ListModelsColdSpawnsPi covers the dropdown-on-connect case:
// the GUI issues list-models before any chat traffic. Without cold-spawn
// behaviour the worker would error with "no active session", leaving the
// model picker stuck on "unknown" until the user sent a prompt.
func TestWorker_ListModelsColdSpawnsPi(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	if w.IsActive() {
		t.Fatal("precondition: worker should be cold")
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	go w.Run(ctx)

	models, err := w.ListModels(ctx)
	if err != nil {
		t.Fatalf("ListModels on cold worker: %v", err)
	}

	if len(models) == 0 {
		t.Fatal("cold spawn produced empty model list")
	}

	if !w.IsActive() {
		t.Fatal("ListModels did not leave pi alive")
	}
}

// Regression: clicking the panel's "reset" button (sends !restart) used
// to drop the user's previously-selected model because processSetModel
// only updated the running pi via RPC — it didn't touch w.piCfg, so the
// next ensurePi(fresh=true) spawn rebuilt args from the configured
// default. Persist the selection on the worker so restarts respect it.
func TestWorker_SetModelPersistsAcrossRestart(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	go w.Run(ctx)

	model, err := w.SetModel(ctx, "local", "gpt-oss")
	if err != nil {
		t.Fatalf("SetModel: %v", err)
	}
	if model == nil || model.Provider != "local" || model.ID != "gpt-oss" {
		t.Fatalf("SetModel returned %+v, want provider=local id=gpt-oss", model)
	}

	w.mu.Lock()
	gotProvider, gotModel := w.piCfg.Provider, w.piCfg.Model
	w.mu.Unlock()

	if gotProvider != "local" || gotModel != "gpt-oss" {
		t.Fatalf("piCfg after SetModel = (%q, %q), want (local, gpt-oss); "+
			"selection is lost on the next restart", gotProvider, gotModel)
	}
}
