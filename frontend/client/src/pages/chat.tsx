import { useState, useRef, useEffect } from "react";
import { ChevronDown } from "lucide-react";
import ChatHeader from "@/components/ChatHeader";
import MessageBubble, { type Message } from "@/components/MessageBubble";
import TypingIndicator from "@/components/TypingIndicator";
import ChatInput from "@/components/ChatInput";
import EmptyState from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import type { SourceType } from "@/components/SourceBadge";

const API_BASE_URL = "/api";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom);
    }
  };

  useEffect(() => {
    scrollToBottom();
    
    // Generate suggestions from recent user questions
    if (messages.length > 0) {
      const userQuestions = messages
        .filter(msg => msg.role === "user")
        .map(msg => msg.content)
        .slice(-3);
      setSuggestions(userQuestions);
    }
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    const assistantMessageId = `assistant-${Date.now()}`;
    const streamingMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, streamingMessage]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: content }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from server");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = "";
      let source: SourceType = "knowledge_base";
      let finalMessageId = assistantMessageId;

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.content) {
                  accumulatedContent += data.content;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    )
                  );
                }

                if (data.tool_used) {
                  if (data.tool_used === "MCP") {
                    source = "web_search";
                  } else if (data.tool_used === "Knowledge Base") {
                    source = "knowledge_base";
                  }
                } else if (data.route) {
                  if (data.route === "web_search") {
                    source = "web_search";
                  } else if (data.route === "guardrail" || data.guardrail_block) {
                    source = "guardrail_block";
                  }
                }

                if (data.message_id) {
                  finalMessageId = data.message_id;
                }

                if (data.done) {
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { 
                            ...msg, 
                            id: finalMessageId,
                            content: accumulatedContent || data.answer || msg.content,
                            source,
                            isStreaming: false 
                          }
                        : msg
                    )
                  );
                }
              } catch (e) {
                console.error("Error parsing SSE data:", e);
              }
            }
          }
        }
      } else {
        const data = await response.json();

        if (data.route === "web_search") {
          source = "web_search";
        } else if (data.guardrail_block || data.route === "guardrail") {
          source = "guardrail_block";
        }

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? {
                  ...msg,
                  id: data.message_id || assistantMessageId,
                  content: data.answer || data.response || "I couldn't generate a response.",
                  source,
                  isStreaming: false,
                }
              : msg
          )
        );
      }
    } catch (error) {
      console.error("Error sending message:", error);
      
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantMessageId));
      
      toast({
        title: "Connection Error",
        description: "Make sure the backend server is running on localhost:8000.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitFeedback = async (
    messageId: string,
    rating: "positive" | "negative",
    comment?: string
  ) => {
    try {
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message_id: messageId,
          rating,
          comment: comment || "",
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to submit feedback");
      }

      // Parse the JSON response to get the updated answer (if any)
      const data = await response.json() as { success: boolean; updated_answer?: string | null };

      // If the backend returns a revised answer, update the corresponding message
      if (data && typeof data.updated_answer === "string" && data.updated_answer.trim().length > 0) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId ? { ...msg, content: data.updated_answer } : msg
          )
        );
      }

      toast({
        title: "Feedback submitted",
        description: "Thank you for helping us improve!",
      });
    } catch (error) {
      console.error("Error submitting feedback:", error);
      toast({
        title: "Error",
        description: "Failed to submit feedback. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    toast({
      title: "Chat cleared",
      description: "Conversation history has been cleared.",
    });
  };

  const handleExampleClick = (question: string) => {
    handleSendMessage(question);
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-b from-background to-muted/20">
      <ChatHeader onClearChat={messages.length > 0 ? handleClearChat : undefined} />

      <div
        ref={chatContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto"
      >
        {messages.length === 0 ? (
          <EmptyState onExampleClick={handleExampleClick} suggestions={suggestions} />
        ) : (
          <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6 space-y-4">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onSubmitFeedback={message.role === "assistant" ? handleSubmitFeedback : undefined}
              />
            ))}
            {isLoading && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {showScrollButton && (
        <Button
          onClick={scrollToBottom}
          size="icon"
          className="fixed bottom-32 sm:bottom-28 right-4 sm:right-6 h-10 w-10 rounded-full shadow-lg z-10"
          data-testid="button-scroll-to-bottom"
        >
          <ChevronDown className="w-5 h-5" />
        </Button>
      )}

      <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
}
