import SourceBadge from '../SourceBadge';

export default function SourceBadgeExample() {
  return (
    <div className="flex flex-wrap gap-2 p-4">
      <SourceBadge source="knowledge_base" />
      <SourceBadge source="web_search" />
      <SourceBadge source="guardrail_block" />
    </div>
  );
}
