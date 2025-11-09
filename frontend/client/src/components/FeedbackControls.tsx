import { useState } from "react";
import { ThumbsUp, ThumbsDown, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface FeedbackControlsProps {
  messageId: string;
  onSubmitFeedback: (messageId: string, rating: "positive" | "negative", comment?: string) => void;
}

export default function FeedbackControls({ messageId, onSubmitFeedback }: FeedbackControlsProps) {
  const [rating, setRating] = useState<"positive" | "negative" | null>(null);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleRatingClick = (newRating: "positive" | "negative") => {
    if (rating === newRating) {
      setRating(null);
      setShowComment(false);
      setSubmitted(false);
      return;
    }
    
    setRating(newRating);
    onSubmitFeedback(messageId, newRating, undefined);
    setSubmitted(true);
    setShowComment(true);
    
    setTimeout(() => {
      setSubmitted(false);
    }, 2000);
  };

  const handleSubmitComment = () => {
    if (rating && comment.trim()) {
      onSubmitFeedback(messageId, rating, comment);
      setShowComment(false);
      setComment("");
      setSubmitted(true);
      setTimeout(() => {
        setSubmitted(false);
      }, 2000);
    }
  };

  return (
    <div className="space-y-2.5" data-testid={`feedback-controls-${messageId}`}>
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => handleRatingClick("positive")}
          data-testid="button-thumbs-up"
          className={`h-8 w-8 p-0 rounded-full transition-colors ${rating === "positive" ? "text-emerald-600 dark:text-emerald-400 bg-emerald-100 dark:bg-emerald-900/30" : "hover:bg-muted"}`}
        >
          <ThumbsUp className={`w-3.5 h-3.5 ${rating === "positive" ? "fill-current" : ""}`} />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => handleRatingClick("negative")}
          data-testid="button-thumbs-down"
          className={`h-8 w-8 p-0 rounded-full transition-colors ${rating === "negative" ? "text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30" : "hover:bg-muted"}`}
        >
          <ThumbsDown className={`w-3.5 h-3.5 ${rating === "negative" ? "fill-current" : ""}`} />
        </Button>
        {submitted && (
          <div className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400 ml-2 animate-in fade-in" data-testid="text-feedback-submitted">
            <CheckCircle className="w-3.5 h-3.5" />
            Thanks for your feedback!
          </div>
        )}
      </div>
      {showComment && (
        <div className="space-y-2 animate-in slide-in-from-top-2">
          <Textarea
            placeholder="Add more details (optional)..."
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            className="resize-none text-sm rounded-xl"
            rows={2}
            data-testid="textarea-feedback-comment"
          />
          <Button
            size="sm"
            onClick={handleSubmitComment}
            disabled={!comment.trim()}
            data-testid="button-submit-feedback"
            className="h-8"
          >
            Submit Comment
          </Button>
        </div>
      )}
    </div>
  );
}
