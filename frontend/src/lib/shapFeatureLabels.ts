const DISPLAY_LABELS: Record<string, string> = {
  approved_cost_cr: 'Approved Project Cost',
  revised_cost_cr: 'Revised Project Cost',
  physical_progress: 'Physical Progress',
  physical_progress_pct: 'Physical Progress',
  financial_progress: 'Financial Progress',
  expenditure_ratio: 'Expenditure Relative to Approved Cost',
  duration_ratio: 'Elapsed Time Relative to Planned Duration',
  elapsed_duration_days: 'Elapsed Project Duration',
  planned_duration_days: 'Planned Project Duration',
  planned_completion_date: 'Planned Completion Date',
  snapshot_date: 'Reporting Snapshot Date',
  schedule_slippage_days: 'Existing Schedule Slippage',
  cost_escalation_percentage: 'Recorded Cost Escalation',
  cumulative_expenditure_cr: 'Cumulative Expenditure',
  implementing_agency: 'Implementing Agency',
  agency_average_delay: 'Agency Average Delay',
  exp12_spend_vs_expected_progress_gap: 'Spend vs Expected Progress Gap',
  exp34_cumulative_abs_cost_revision_pct: 'Cumulative Cost Revision',
  exp34_max_cost_escalation: 'Maximum Cost Escalation',
  sector: 'Project Sector',
  ministry: 'Line Ministry',
};

const EXPLANATION_SUBJECTS: Record<string, string> = {
  planned_completion_date: 'A later planned completion date',
  exp34_cumulative_abs_cost_revision_pct: 'Cumulative cost revisions',
  exp34_max_cost_escalation: 'Maximum cost escalation',
};

const TOKEN_LABELS: Record<string, string> = {
  abs: 'Absolute', avg: 'Average', cr: '₹ Crore', pct: 'Percentage',
};

const SMALL_WORDS = new Set(['and', 'at', 'for', 'of', 'to', 'vs']);

function titleCase(value: string): string {
  return value.split(' ').map((word, index) => {
    const lower = word.toLowerCase();
    if (index > 0 && SMALL_WORDS.has(lower)) return lower;
    return TOKEN_LABELS[lower] ?? `${lower[0]?.toUpperCase() ?? ''}${lower.slice(1)}`;
  }).join(' ');
}

/** Presentation-only labels for API feature identifiers; never used for inference. */
export function shapFeatureLabel(feature: string): string {
  if (DISPLAY_LABELS[feature]) return DISPLAY_LABELS[feature];
  return titleCase(feature.replace(/^exp\d+_/i, '').replaceAll('_', ' '));
}

/** Grammar-aware subject for non-causal local-contribution copy. */
export function shapExplanationSubject(feature: string): string {
  return EXPLANATION_SUBJECTS[feature] ?? shapFeatureLabel(feature);
}
