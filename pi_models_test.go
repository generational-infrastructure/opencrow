package main

import (
	"reflect"
	"testing"
)

// TestPiProcess_ListModels verifies that ListModels parses pi's
// get_available_models response into a ModelInfo slice and merges the
// Active flag from get_state.
func TestPiProcess_ListModels(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	// Spawn pi via sendWithRetry so the process is live.
	if _, _, err := w.sendWithRetry(t.Context(), "hello", nil); err != nil {
		t.Fatalf("sendWithRetry: %v", err)
	}

	pi := w.pi
	if pi == nil || !pi.IsAlive() {
		t.Fatal("pi not alive after initial prompt")
	}

	models, err := pi.ListModels(t.Context())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}

	want := []ModelInfo{
		{Provider: "local", ID: "qwen3", ContextWindow: 32768, Reasoning: false, Active: true},
		{Provider: "local", ID: "gpt-oss", ContextWindow: 131072, Reasoning: true},
	}

	if !reflect.DeepEqual(models, want) {
		t.Fatalf("ListModels = %+v, want %+v", models, want)
	}
}

// TestPiProcess_SetModel verifies that SetModel sends the right command
// shape (provider + modelId) and parses the response.
func TestPiProcess_SetModel(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	if _, _, err := w.sendWithRetry(t.Context(), "hello", nil); err != nil {
		t.Fatalf("sendWithRetry: %v", err)
	}

	pi := w.pi
	if pi == nil || !pi.IsAlive() {
		t.Fatal("pi not alive after initial prompt")
	}

	model, err := pi.SetModel(t.Context(), "local", "gpt-oss")
	if err != nil {
		t.Fatalf("SetModel: %v", err)
	}

	if model == nil {
		t.Fatal("SetModel returned nil model")
	}

	if model.Provider != "local" || model.ID != "gpt-oss" {
		t.Fatalf("SetModel = %+v, want provider=local id=gpt-oss", *model)
	}
}

// TestPiProcess_ListModels_DeadProcess verifies that ListModels fails
// fast with a meaningful error when pi is not alive.
func TestPiProcess_ListModels_DeadProcess(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	// Spawn then kill so we have a non-nil but dead PiProcess.
	if _, _, err := w.sendWithRetry(t.Context(), "hello", nil); err != nil {
		t.Fatalf("sendWithRetry: %v", err)
	}

	pi := w.pi
	pi.Kill()

	if _, err := pi.ListModels(t.Context()); err == nil {
		t.Fatal("ListModels on dead process should return error")
	}
}

// TestPiProcess_SetModel_DeadProcess verifies SetModel fails fast on dead pi.
func TestPiProcess_SetModel_DeadProcess(t *testing.T) {
	t.Parallel()

	w := newFakePiWorker(t)

	if _, _, err := w.sendWithRetry(t.Context(), "hello", nil); err != nil {
		t.Fatalf("sendWithRetry: %v", err)
	}

	pi := w.pi
	pi.Kill()

	if _, err := pi.SetModel(t.Context(), "local", "qwen3"); err == nil {
		t.Fatal("SetModel on dead process should return error")
	}
}
