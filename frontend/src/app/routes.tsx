import { Navigate, Route, Routes } from 'react-router-dom';
import { Dashboard } from '../pages/Dashboard';
import { Projects } from '../pages/Projects';
import { ProjectDetail } from '../pages/ProjectDetail';
import { EarlyWarnings } from '../pages/EarlyWarnings';
import { PredictionAccuracyPage } from '../pages/PredictionAccuracyPage';
import { DashboardLayout } from '../components/layout/DashboardLayout';

export function AppRoutes() {
  return <DashboardLayout><Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/projects" element={<Projects />} />
    <Route path="/projects/:projectId" element={<ProjectDetail />} />
    <Route path="/early-warnings" element={<EarlyWarnings />} />
    <Route path="/prediction-accuracy" element={<PredictionAccuracyPage />} />
    <Route path="*" element={<Navigate to="/dashboard" replace />} />
  </Routes></DashboardLayout>;
}
