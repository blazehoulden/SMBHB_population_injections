import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_population_presets_and_paths(self):
        sys.path.insert(0, str(ROOT))
        import config

        self.assertIn("test", config.POPULATION_CONFIGS)
        self.assertTrue(config.PAR_DIR.is_absolute())
        self.assertTrue(config.TIM_DIR.is_absolute())
        self.assertTrue(config.NOISEFILE.is_absolute())

    def test_environment_path_overrides(self):
        env = os.environ.copy()
        env.update(
            {
                "SMBHB_PAR_DIR": "/tmp/smbhb-par",
                "SMBHB_TIM_DIR": "/tmp/smbhb-tim",
                "SMBHB_NOISE_FILE": "/tmp/smbhb-noise.json",
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import config; "
                    "assert str(config.PAR_DIR) == '/tmp/smbhb-par'; "
                    "assert str(config.TIM_DIR) == '/tmp/smbhb-tim'; "
                    "assert str(config.NOISEFILE) == '/tmp/smbhb-noise.json'"
                ),
            ],
            cwd=ROOT,
            env=env,
            check=False,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
