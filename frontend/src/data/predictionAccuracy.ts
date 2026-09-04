export type RiskLevel = "Low" | "Moderate" | "High" | "Critical";
export type EvaluationConfig = {
  rule: string;
  trainingPeriod: string;
  holdoutPeriod: string;
  method: string;
  status: "pending" | "running" | "complete" | "failed";
};
export type PredictionEvidence = {
  id: string;
  name: string;
  sector: string;
  actualCostOverrun: number;
  predictedCostOverrun: number;
  actualDelay: number;
  predictedDelay: number;
  actualRisk: RiskLevel;
  predictedRisk: RiskLevel;
  earlyWarning: RiskLevel;
  confidence: string;
};

export const defaultConfig: EvaluationConfig = {
  rule: "Default Lifecycle Run",
  trainingPeriod: "2008 – 2018",
  holdoutPeriod: "2019 – 2020",
  method: "Out-of-time",
  status: "complete",
};
export const evaluationOptions = {
  rules: [
    "Default Lifecycle Run",
    "Cost Validation Run",
    "Schedule Validation Run",
    "Risk Classification Run",
    "Rolling Validation",
  ],
  training: ["2008 – 2018", "2009 – 2019", "2010 – 2020"],
  holdout: ["2019 – 2020", "2020 – 2021", "2021 – 2022"],
  methods: ["Out-of-time", "Holdout", "Cross-validation", "Rolling-window"],
};

export const predictionEvidence: PredictionEvidence[] = [
  {
    id: "P-1023",
    name: "Mumbai Metro-3",
    sector: "Urban Transit",
    actualCostOverrun: 26.7,
    predictedCostOverrun: 31.4,
    actualDelay: 423,
    predictedDelay: 470,
    actualRisk: "Critical",
    predictedRisk: "Critical",
    earlyWarning: "High",
    confidence: "0.82 (±1.4)",
  },
  {
    id: "P-0876",
    name: "River Valley Highway",
    sector: "Roads",
    actualCostOverrun: 18.3,
    predictedCostOverrun: 15.2,
    actualDelay: 290,
    predictedDelay: 310,
    actualRisk: "High",
    predictedRisk: "High",
    earlyWarning: "High",
    confidence: "0.74 (±1.6)",
  },
  {
    id: "P-0651",
    name: "Baroda Ring Road",
    sector: "Roads",
    actualCostOverrun: 9.7,
    predictedCostOverrun: 11.1,
    actualDelay: 168,
    predictedDelay: 150,
    actualRisk: "Moderate",
    predictedRisk: "Moderate",
    earlyWarning: "Moderate",
    confidence: "0.63 (±1.8)",
  },
  {
    id: "P-0432",
    name: "Solar Park Rajasthan",
    sector: "Power",
    actualCostOverrun: 5.2,
    predictedCostOverrun: 4.8,
    actualDelay: 87,
    predictedDelay: 82,
    actualRisk: "Low",
    predictedRisk: "Low",
    earlyWarning: "Low",
    confidence: "0.81 (±1.2)",
  },
  {
    id: "P-0218",
    name: "Water Supply Project",
    sector: "Water",
    actualCostOverrun: 3.1,
    predictedCostOverrun: 2.9,
    actualDelay: 46,
    predictedDelay: 41,
    actualRisk: "Low",
    predictedRisk: "Low",
    earlyWarning: "Low",
    confidence: "0.88 (±1.1)",
  },
  {
    id: "P-1107",
    name: "Eastern Freight Corridor",
    sector: "Railways",
    actualCostOverrun: 43.2,
    predictedCostOverrun: 38.7,
    actualDelay: 762,
    predictedDelay: 701,
    actualRisk: "Critical",
    predictedRisk: "High",
    earlyWarning: "Critical",
    confidence: "0.77 (±2.0)",
  },
  {
    id: "P-1162",
    name: "Coastal Port Modernisation",
    sector: "Ports",
    actualCostOverrun: 14.4,
    predictedCostOverrun: 19.3,
    actualDelay: 225,
    predictedDelay: 276,
    actualRisk: "High",
    predictedRisk: "High",
    earlyWarning: "Moderate",
    confidence: "0.70 (±1.7)",
  },
  {
    id: "P-0913",
    name: "North Grid Reinforcement",
    sector: "Power",
    actualCostOverrun: 32.8,
    predictedCostOverrun: 35.5,
    actualDelay: 548,
    predictedDelay: 524,
    actualRisk: "Critical",
    predictedRisk: "Critical",
    earlyWarning: "High",
    confidence: "0.84 (±1.3)",
  },
];

export const confusionMatrix = [
  { actual: "Low", values: [893, 98, 17, 2] },
  { actual: "Moderate", values: [180, 743, 59, 18] },
  { actual: "High", values: [61, 191, 662, 86] },
  { actual: "Critical", values: [18, 42, 153, 787] },
] as const;
export const riskLabels: RiskLevel[] = ["Low", "Moderate", "High", "Critical"];
export const expandingWindow = [
  { year: "2016", cost: 7.2, delay: 63 },
  { year: "2017", cost: 8.4, delay: 55 },
  { year: "2018", cost: 7.9, delay: 58 },
  { year: "2019", cost: 9.1, delay: 49 },
  { year: "2020", cost: 8.4, delay: 47.6 },
];
export const shapFeatures = [
  {
    feature: "Physical Progress",
    value: 0.39,
    interpretation: "Execution pace is the strongest leading signal.",
  },
  {
    feature: "Expenditure Progress",
    value: 0.28,
    interpretation: "Spend-to-work variance indicates exposure.",
  },
  {
    feature: "Cost Overrun (Past)",
    value: 0.18,
    interpretation: "Historic cost deviation persists over time.",
  },
  {
    feature: "Variance Slippage",
    value: 0.12,
    interpretation: "Milestone variance contributes to delay risk.",
  },
  {
    feature: "Inflation Variance",
    value: 0.09,
    interpretation: "Input-price movements affect final cost.",
  },
];

export const shiftEvidence = (config: EvaluationConfig) => {
  const shift =
    (config.rule === defaultConfig.rule ? 0 : 1.2) +
    (config.method === "Rolling-window" ? -0.6 : 0) +
    (config.trainingPeriod === "2010 – 2020" ? 0.4 : 0);
  return predictionEvidence.map((project, index) => ({
    ...project,
    predictedCostOverrun: Number(
      (project.predictedCostOverrun + shift * (index % 2 ? -1 : 1)).toFixed(1),
    ),
    predictedDelay: Math.round(
      project.predictedDelay + shift * 8 * (index % 2 ? -1 : 1),
    ),
  }));
};
export const histogram = (
  rows: PredictionEvidence[],
  kind: "cost" | "delay",
) => {
  const buckets =
    kind === "cost"
      ? [
          { label: "0–5", min: 0, max: 5 },
          { label: "5–10", min: 5, max: 10 },
          { label: "10–20", min: 10, max: 20 },
          { label: "20–30", min: 20, max: 30 },
          { label: "30+", min: 30, max: Infinity },
        ]
      : [
          { label: "0–5", min: 0, max: 5 },
          { label: "5–15", min: 5, max: 15 },
          { label: "15–30", min: 15, max: 30 },
          { label: "30–60", min: 30, max: 60 },
          { label: ">60", min: 60, max: Infinity },
        ];
  return buckets.map((bucket, index) => ({
    bucket: bucket.label,
    frequency: rows.filter((row) => {
      const value = Math.abs(
        kind === "cost"
          ? row.predictedCostOverrun - row.actualCostOverrun
          : row.predictedDelay - row.actualDelay,
      );
      return index === 0
        ? value <= bucket.max
        : value > bucket.min && value <= bucket.max;
    }).length,
  }));
};
export const matrixMetrics = () => {
  const total = confusionMatrix.reduce(
    (sum, row) => sum + row.values.reduce((a, b) => a + b, 0),
    0,
  );
  const correct = confusionMatrix.reduce(
    (sum, row, index) => sum + row.values[index],
    0,
  );
  return { accuracy: correct / total, macroF1: 0.72 };
};
