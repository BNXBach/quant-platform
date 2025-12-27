def main() -> None:
    from backend.quant.phase1_cvar import data, generate_scenarios, optimize, optimize_stability, run_test

    data.main()
    generate_scenarios.main()
    optimize.main()
    optimize_stability.main()
    run_test.main()

if __name__ == "__main__":
    main()