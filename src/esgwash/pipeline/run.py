"""CLI: python -m esgwash.pipeline.run --stage all|build_corpus|...|index"""
import argparse

from esgwash.pipeline.stages import STAGES, run_stage


def main(argv=None):
    p = argparse.ArgumentParser(description="esgwash end-to-end pipeline")
    p.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = p.parse_args(argv)
    # TODO(Phase A1): load config tung stage, chay tuan tu
    raise NotImplementedError


if __name__ == "__main__":
    main()
