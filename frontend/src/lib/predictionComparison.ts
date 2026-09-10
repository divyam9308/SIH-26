export function predictionActualRatio(predicted: number | null | undefined, actual: number | null | undefined) {
  if (predicted == null || actual == null || actual === 0) return null;
  return (predicted / actual) * 100;
}

export function predictionActualError(predicted: number | null | undefined, actual: number | null | undefined) {
  const ratio = predictionActualRatio(predicted, actual);
  return ratio === null ? null : Math.abs(100 - ratio);
}

export function predictionActualLabel(predicted: number | null | undefined, actual: number | null | undefined) {
  const error = predictionActualError(predicted, actual);
  return error === null ? 'N/A' : `${error.toFixed(1)}% error`;
}
