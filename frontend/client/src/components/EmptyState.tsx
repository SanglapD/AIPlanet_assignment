import { Calculator, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface EmptyStateProps {
  onExampleClick?: (question: string) => void;
  suggestions?: string[];
}

export default function EmptyState({ onExampleClick, suggestions = [] }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 py-12 animate-in fade-in duration-500" data-testid="empty-state">
      <div className="max-w-lg text-center space-y-8">
        <div className="flex justify-center">
          <div className="relative">
            <div className="flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-primary to-primary/80 text-primary-foreground shadow-lg">
              <Calculator className="w-10 h-10" />
            </div>
            <div className="absolute -top-1 -right-1">
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-emerald-600 text-white shadow-md">
                <Sparkles className="w-4 h-4" />
              </div>
            </div>
          </div>
        </div>
        
        <div className="space-y-3">
          <h2 className="text-3xl font-bold tracking-tight" data-testid="text-welcome-heading">
            Math Tutor AI
          </h2>
          <p className="text-base text-muted-foreground leading-relaxed">
            Ask any mathematical question and receive detailed, step-by-step solutions powered by intelligent routing between knowledge base and web search.
          </p>
        </div>

        {suggestions.length > 0 && (
          <div className="space-y-4 pt-2">
            <div className="flex items-center justify-center gap-2">
              <div className="h-px flex-1 bg-border" />
              <p className="text-sm font-medium text-muted-foreground">Suggested questions</p>
              <div className="h-px flex-1 bg-border" />
            </div>
            <div className="flex flex-wrap gap-2 justify-center">
              {suggestions.map((question, idx) => (
                <Badge
                  key={idx}
                  variant="secondary"
                  className="cursor-pointer hover-elevate active-elevate-2 text-sm py-1.5 px-3"
                  onClick={() => onExampleClick?.(question)}
                  data-testid={`badge-example-${idx}`}
                >
                  {question}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
