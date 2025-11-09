import { BookOpen, Search, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export type SourceType = "knowledge_base" | "web_search" | "guardrail_block";

interface SourceBadgeProps {
  source: SourceType;
}

const sourceConfig = {
  knowledge_base: {
    label: "Knowledge Base",
    icon: BookOpen,
    className: "bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 dark:border-emerald-800",
  },
  web_search: {
    label: "Web Search",
    icon: Search,
    className: "bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800",
  },
  guardrail_block: {
    label: "Blocked",
    icon: ShieldAlert,
    className: "bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800",
  },
};

export default function SourceBadge({ source }: SourceBadgeProps) {
  const config = sourceConfig[source];
  const Icon = config.icon;

  return (
    <Badge
      variant="outline"
      className={`gap-1.5 text-xs font-medium ${config.className}`}
      data-testid={`badge-source-${source}`}
    >
      <Icon className="w-3 h-3" />
      {config.label}
    </Badge>
  );
}
