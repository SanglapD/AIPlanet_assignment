export default function TypingIndicator() {
  return (
    <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2 duration-300" data-testid="typing-indicator">
      <div className="max-w-3xl">
        <div className="rounded-3xl rounded-tl-md px-5 py-3.5 bg-card/50 backdrop-blur-sm border border-card-border/50 shadow-sm">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
            <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
            <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
          </div>
        </div>
      </div>
    </div>
  );
}
