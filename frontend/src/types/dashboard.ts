export interface Project { id:string | number; code?:string; name:string; sector:string; riskLevel?:string; riskScore:number; costRisk:number; scheduleRisk:number; progress:number | null; warning:string }
export interface DashboardKPI { title:string; value:string; change:string; tone:'blue'|'red'|'orange' }
export interface RiskDistribution { name:string; value:number; color:string }
export interface ExpenditureProgressPoint { x:number; y:number; group:'On Track'|'Monitor'|'At Risk' }
export interface WarningDriver { name:string; value:number }
export interface RiskTrendPoint { month:string; value:number }
export interface DashboardData {
  kpis:DashboardKPI[]; projects:Project[]; riskDistribution:RiskDistribution[]; totalProjects?:number;
  expenditureProgress:ExpenditureProgressPoint[]; warningDrivers:WarningDriver[]; riskTrend?:RiskTrendPoint[];
  riskTrendStatus?:string; modelVersion?:string | null; datasetSnapshot?:string | null; modelScope?:string | null; inferenceTimestamp?:string;
}
