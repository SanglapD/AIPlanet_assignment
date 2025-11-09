import MathContent from "./MathContent";
import SourceBadge, { type SourceType } from "./SourceBadge";
import FeedbackControls from "./FeedbackControls";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  source?: SourceType;
  timestamp: Date;
  isStreaming?: boolean;
}

interface MessageBubbleProps {
  message: Message;
  onSubmitFeedback?: (messageId: string, rating: "positive" | "negative", comment?: string) => void;
}

export default function MessageBubble({ message, onSubmitFeedback }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"} animate-in fade-in slide-in-from-bottom-2 duration-300`}
      data-testid={`message-${message.id}`}
    >
      <div className={`w-full ${isUser ? "max-w-2xl" : "max-w-3xl"}`}>
        <div
          className={`rounded-3xl px-5 py-3.5 shadow-sm ${
            isUser
              ? "bg-gradient-to-br from-primary to-primary/90 text-primary-foreground rounded-tr-md"
              : "bg-card/50 backdrop-blur-sm border border-card-border/50 rounded-tl-md"
          }`}
        >
          {!isUser && message.source && (
            <div className="mb-3">
              <SourceBadge source={message.source} />
            </div>
          )}
          <div className={`${isUser ? "text-[15px] leading-relaxed" : "text-[15px] leading-relaxed"}`}>
            {isUser ? (
              <p data-testid="text-user-message" className="whitespace-pre-wrap">{message.content}</p>
            ) : (
              <MathContent content={message.content} className="leading-loose" />
            )}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 ml-1 bg-foreground/60 animate-pulse" />
            )}
          </div>
        </div>
        <div className="mt-1.5 px-3 text-xs text-muted-foreground">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
        {!isUser && onSubmitFeedback && !message.isStreaming && (
          <div className="mt-2 px-3">
            <FeedbackControls messageId={message.id} onSubmitFeedback={onSubmitFeedback} />
          </div>
        )}
      </div>
    </div>
  );
}
