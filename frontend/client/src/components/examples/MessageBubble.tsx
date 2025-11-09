import MessageBubble from '../MessageBubble';

export default function MessageBubbleExample() {
  const userMessage = {
    id: "msg-1",
    role: "user" as const,
    content: "What is the derivative of $f(x) = x^2 + 3x + 2$?",
    timestamp: new Date(),
  };

  const assistantMessage = {
    id: "msg-2",
    role: "assistant" as const,
    content: `To find the derivative of $f(x) = x^2 + 3x + 2$, we apply the power rule.

**Step 1:** Apply the power rule to each term
- $(x^2)' = 2x$
- $(3x)' = 3$
- $(2)' = 0$

**Step 2:** Combine the results

$$f'(x) = 2x + 3$$

Therefore, the derivative is $f'(x) = 2x + 3$.`,
    source: "knowledge_base" as const,
    timestamp: new Date(),
  };

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <MessageBubble message={userMessage} />
      <MessageBubble
        message={assistantMessage}
        onSubmitFeedback={(id, rating, comment) => {
          console.log('Feedback:', { id, rating, comment });
        }}
      />
    </div>
  );
}
