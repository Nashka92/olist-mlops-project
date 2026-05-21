from pathlib import Path


def test_artifacts_in_gitignore():
    gitignore = Path(".gitignore").read_text()

    assert "artifacts/" in gitignore or "*.joblib" in gitignore


def test_data_in_gitignore():
    gitignore = Path(".gitignore").read_text()

    assert "data/" in gitignore or "*.csv" in gitignore


def test_no_training_code_in_prediction():
    for file in Path("src/ml_olist/prediction").rglob("*.py"):
        content = file.read_text()

        assert "train_test_split" not in content


def test_env_file_not_committed():
    gitignore = Path(".gitignore").read_text()

    assert ".env" in gitignore
