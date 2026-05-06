package socket

import (
	"context"
	"errors"
	"net"
	"reflect"
	"testing"

	"github.com/pinpox/opencrow/backend"
)

// Test fixtures used across model tests.
const (
	testProvider = "local"
	testModelID  = "qwen3"
)

// stubModelService is a test double for the ModelService interface.
type stubModelService struct {
	models     []backend.ModelInfo
	listErr    error
	setErr     error
	lastSet    *backend.ModelInfo
	setCallArg struct {
		provider string
		modelID  string
	}
}

func (s *stubModelService) ListModels(_ context.Context) ([]backend.ModelInfo, error) {
	if s.listErr != nil {
		return nil, s.listErr
	}

	return s.models, nil
}

func (s *stubModelService) SetModel(_ context.Context, provider, modelID string) (*backend.ModelInfo, error) {
	s.setCallArg.provider = provider
	s.setCallArg.modelID = modelID

	if s.setErr != nil {
		return nil, s.setErr
	}

	if s.lastSet != nil {
		return s.lastSet, nil
	}

	return &backend.ModelInfo{Provider: provider, ID: modelID, ContextWindow: 4096}, nil
}

func TestListModels_ReturnsModels(t *testing.T) {
	t.Parallel()

	svc := &stubModelService{
		models: []backend.ModelInfo{
			{Provider: testProvider, ID: testModelID, ContextWindow: 32768, Reasoning: false, Active: true},
			{Provider: testProvider, ID: "gpt-oss", ContextWindow: 131072, Reasoning: true},
		},
	}

	_, sockPath, cancel := startBackend(t, nil, svc)
	defer cancel()

	conn := dial(t, sockPath)
	defer conn.Close()

	sendCommand(t, conn, command{Cmd: cmdListModels})

	ev := readEvent(t, conn)
	if ev.Kind != evModels {
		t.Fatalf("kind = %q, want %q", ev.Kind, evModels)
	}

	want := []backend.ModelInfo{
		{Provider: testProvider, ID: testModelID, ContextWindow: 32768, Reasoning: false, Active: true},
		{Provider: testProvider, ID: "gpt-oss", ContextWindow: 131072, Reasoning: true},
	}

	if !reflect.DeepEqual(ev.Models, want) {
		t.Fatalf("models = %+v, want %+v", ev.Models, want)
	}
}

// TestListModels_ServiceErrorEmitsError: when the service rejects the
// request (pi failed to spawn, RPC error, etc.) the client gets an
// explicit error event so the UI can surface the failure.
func TestListModels_ServiceErrorEmitsError(t *testing.T) {
	t.Parallel()

	svc := &stubModelService{listErr: errors.New("pi crash")}

	_, sockPath, cancel := startBackend(t, nil, svc)
	defer cancel()

	conn := dial(t, sockPath)
	defer conn.Close()

	sendCommand(t, conn, command{Cmd: cmdListModels})

	ev := readEvent(t, conn)
	if ev.Kind != evError {
		t.Fatalf("kind = %q, want %q", ev.Kind, evError)
	}

	if ev.Text == "" {
		t.Fatal("error event should carry a non-empty message")
	}
}

func TestSetModel_ForwardsToService(t *testing.T) {
	t.Parallel()

	svc := &stubModelService{}

	_, sockPath, cancel := startBackend(t, nil, svc)
	defer cancel()

	conn := dial(t, sockPath)
	defer conn.Close()

	sendCommand(t, conn, command{Cmd: cmdSetModel, Provider: testProvider, ModelID: testModelID})

	ev := readEvent(t, conn)
	if ev.Kind != evModels {
		t.Fatalf("kind = %q, want %q", ev.Kind, evModels)
	}

	if len(ev.Models) != 1 || !ev.Models[0].Active {
		t.Fatalf("expected single active model, got %+v", ev.Models)
	}

	if ev.Models[0].Provider != testProvider || ev.Models[0].ID != testModelID {
		t.Fatalf("model = %+v, want provider=local id=qwen3", ev.Models[0])
	}

	if svc.setCallArg.provider != testProvider || svc.setCallArg.modelID != testModelID {
		t.Fatalf("service called with %+v, want provider=local modelID=qwen3", svc.setCallArg)
	}
}

// TestSetModel_ServiceErrorEmitsError: when the service rejects the
// requested model (unknown, invalid, pi RPC error) the requesting
// client gets an error event carrying the underlying message.
func TestSetModel_ServiceErrorEmitsError(t *testing.T) {
	t.Parallel()

	svc := &stubModelService{setErr: errors.New("model not found")}

	_, sockPath, cancel := startBackend(t, nil, svc)
	defer cancel()

	conn := dial(t, sockPath)
	defer conn.Close()

	sendCommand(t, conn, command{Cmd: cmdSetModel, Provider: testProvider, ModelID: "nonexistent"})

	ev := readEvent(t, conn)
	if ev.Kind != evError {
		t.Fatalf("kind = %q, want %q", ev.Kind, evError)
	}

	if ev.Text == "" {
		t.Fatal("error event should carry a non-empty message")
	}
}

// TestBroadcastModels_PushesToAllClients: BroadcastModels sends the
// current list to every connected client, not just one. Used by the
// worker after a fresh pi spawn so dropdowns sync without polling.
func TestBroadcastModels_PushesToAllClients(t *testing.T) {
	t.Parallel()

	svc := &stubModelService{
		models: []backend.ModelInfo{
			{Provider: testProvider, ID: testModelID, ContextWindow: 32768, Active: true},
			{Provider: testProvider, ID: "smollm"},
		},
	}

	b, sockPath, cancel := startBackend(t, nil, svc)
	defer cancel()

	// Two clients, both must observe the broadcast.
	conn1 := dialSynced(t, sockPath)
	defer conn1.Close()

	conn2 := dialSynced(t, sockPath)
	defer conn2.Close()

	b.BroadcastModels(t.Context())

	for i, conn := range []net.Conn{conn1, conn2} {
		ev := readEvent(t, conn)
		if ev.Kind != evModels {
			t.Fatalf("client %d: kind = %q, want %q", i, ev.Kind, evModels)
		}

		if len(ev.Models) != 2 {
			t.Fatalf("client %d: expected 2 models, got %+v", i, ev.Models)
		}
	}
}
