export const SAVED_WINDOW_STORAGE_KEY = 'infrasight-saved-training-window';

export const SAVED_WINDOWS = [
  { key: '2001_2017', from: 2001, to: 2017, dataFrom: 2018, dataTo: 2025 },
  { key: '2001_2021', from: 2001, to: 2021, dataFrom: 2022, dataTo: 2025 },
  { key: '2001_2022', from: 2001, to: 2022, dataFrom: 2023, dataTo: 2025 },
] as const;

export type SavedWindowKey = typeof SAVED_WINDOWS[number]['key'];

export function getSavedWindow(key: string) {
  return SAVED_WINDOWS.find((item) => item.key === key) ?? SAVED_WINDOWS[0];
}

export function getDataRangeLabel(key: string) {
  const selected = getSavedWindow(key);
  return `${selected.dataFrom}–${selected.dataTo}`;
}

export function getTrainingWindowLabel(key: string) {
  const selected = getSavedWindow(key);
  return `Training ${selected.from}–${selected.to} → Data Range ${selected.dataFrom}–${selected.dataTo}`;
}
