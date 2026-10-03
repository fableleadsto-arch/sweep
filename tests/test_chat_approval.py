from sweep_neural_mesh.chat import engine


class FakeBase:
    def ready(self, timeout=0):
        return True

    def generate(self, *args, **kwargs):
        return "RUN: echo approved"


def test_model_commands_do_not_execute_without_approval(monkeypatch):
    chat = engine.SweepChat(enable_shell=True, enable_mesh=False)
    chat._base = FakeBase()
    calls = []
    monkeypatch.setattr(engine, "run_command", lambda command: calls.append(command))
    response = chat._shell_flow("run a terminal command")
    assert not calls
    assert "approval" in response.text


def test_model_commands_execute_only_the_approved_command(monkeypatch):
    approvals, calls = [], []

    def approve(command):
        approvals.append(command)
        return True

    def run(command):
        calls.append(command)
        return engine.CommandResult(ok=True, command=command, output="approved")

    chat = engine.SweepChat(enable_shell=True, enable_mesh=False, confirm_command=approve)
    chat._base = FakeBase()
    monkeypatch.setattr(engine, "run_command", run)
    chat._shell_flow("run a terminal command")
    assert approvals == calls == ["echo approved"]


def test_chat_shell_is_disabled_by_default():
    assert not engine.SweepChat(enable_mesh=False).enable_shell
