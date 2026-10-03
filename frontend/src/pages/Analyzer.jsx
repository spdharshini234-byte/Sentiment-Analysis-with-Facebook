import CommentAnalyzer from "../components/CommentAnalyzer.jsx";
import { PageHeader } from "../components/State.jsx";

export default function Analyzer() {
  return (
    <>
      <PageHeader title="Comment Analyzer" subtitle="Predict the sentiment of a single comment with the trained model." />
      <CommentAnalyzer />
    </>
  );
}
