from __future__ import annotations

import tempfile
import unittest
from pathlib import Path, PurePosixPath

from sandbox_harness.backends import ProotSession, SandboxConfig


class ProotSessionTest(unittest.TestCase):
    def test_uses_exact_workspace_bind_and_no_kill_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            session = ProotSession(
                workspace=workspace,
                config=SandboxConfig(),
                proot_path="printf",
                rootfs=Path("/"),
                mount_path=PurePosixPath("/workspace"),
            )

            result = session.run(["bash", "-lc", "true"])
            argv = list(result.argv)

            bind_index = argv.index("-b")
            cwd_index = argv.index("-w")
            self.assertEqual(argv[bind_index + 1], f"{workspace}:/workspace!")
            self.assertEqual(argv[cwd_index + 1], "/workspace")
            self.assertNotIn("--kill-on-exit", argv)


if __name__ == "__main__":
    unittest.main()
