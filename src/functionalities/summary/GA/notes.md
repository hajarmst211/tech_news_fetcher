#### technique i:
- GridSampler
- Population Size (pop_size): Evaluated as discrete categorical choices: [15, 20, 25].

- Crossover Probability (p_crossover): Ranged from 0.60 to 0.80 with a step size of 0.10 (yielding values: 0.60, 0.70, 0.80).

- Mutation Probability (p_mutation): Ranged from 0.07 to 0.11 with a step size of 0.02 (yielding values: 0.07, 0.09, 0.11)
- Coverage Weight (w_coverage): Used uniformly for the standard, MCBA, and RPM fitness calculations. It ranged from 0.45 to 0.55 with a step size of 0.05 (yielding values: 0.45, 0.50, 0.55).

- MCBA Position Weight (w_position_mcba): Ranged from 0.50 to 0.80 with a step size of 0.15 (yielding values: 0.50, 0.65, 0.80).

- Penalty Weight (penalty_weight): Controlled the strictness of the length constraint when the summary deviated from the target length. It ranged from 0.10 to 1.10 with a step size of 0.50 (yielding values: 0.10, 0.60, 1.10)


- Certain parameters remained constant during the optimization process:

    Generations: Fixed at 20.

    Target Summary Length: Fixed at 5 sentences.

    Redundancy Penalty (Standard): Fixed at 0.3.

    Redundancy Penalty (MCBA): Fixed at 0.2.

    Redundancy Penalty (RPM): Fixed at 0.2

### Best Configuration (Trial 35 out of 79 (forced stop))
Parameter Characteristics:

    Population Size (Pop Size): 25

    Probability of Crossover (P Crossover): 0.800

    Probability of Mutation (P Mutation): 0.070

    Weight Coverage (W Coverage): 0.55

    Weight Positive MCBA (W Pos MCBA): 0.80

    Penalty Weight: 1.10

Resulting Metrics:

    Overall Mean: 0.2232

    Mean Std (Multi-Seed): 0.1985

    Mean MCBA (Multi-Seed): 0.2227

    Mean RPM (Multi-Seed): 0.2485