import BatchAnalyzer from "../components/BatchAnalyzer.jsx";
import { PageHeader } from "../components/State.jsx";

export default function Batch() {
  return (
    <>
      <PageHeader title="Batch Analyzer" subtitle="Upload a CSV, get a sentiment and confidence for every comment, then download the results." />
      <BatchAnalyzer />
    </>
  );
}
