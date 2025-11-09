import { useEffect, useRef } from "react";
import katex from "katex";

interface MathContentProps {
  content: string;
  className?: string;
}

export default function MathContent({ content, className = "" }: MathContentProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const renderMath = () => {
      if (!containerRef.current) return;
      
      const text = content;
      const parts: (string | { type: 'math'; content: string; display: boolean })[] = [];
      let lastIndex = 0;

      const displayMathRegex = /\$\$([\s\S]+?)\$\$/g;
      const inlineMathRegex = /\$([^\$\n]+?)\$/g;

      let match: RegExpExecArray | null;
      const matches: Array<{ index: number; length: number; content: string; display: boolean }> = [];

      while ((match = displayMathRegex.exec(text)) !== null) {
        matches.push({
          index: match.index,
          length: match[0].length,
          content: match[1],
          display: true,
        });
      }

      while ((match = inlineMathRegex.exec(text)) !== null) {
        const matchIndex = match.index;
        const matchLength = match[0].length;
        const matchContent = match[1];
        const isPartOfDisplay = matches.some(
          m => matchIndex >= m.index && matchIndex < m.index + m.length
        );
        if (!isPartOfDisplay) {
          matches.push({
            index: matchIndex,
            length: matchLength,
            content: matchContent,
            display: false,
          });
        }
      }

      matches.sort((a, b) => a.index - b.index);

      matches.forEach(m => {
        if (m.index > lastIndex) {
          parts.push(text.substring(lastIndex, m.index));
        }
        parts.push({ type: 'math', content: m.content, display: m.display });
        lastIndex = m.index + m.length;
      });

      if (lastIndex < text.length) {
        parts.push(text.substring(lastIndex));
      }

      containerRef.current.innerHTML = '';
      
      parts.forEach(part => {
        if (typeof part === 'string') {
          const lines = part.split('\n');
          lines.forEach((line, i) => {
            if (i > 0) {
              containerRef.current?.appendChild(document.createElement('br'));
            }
            if (line) {
              containerRef.current?.appendChild(document.createTextNode(line));
            }
          });
        } else {
          const span = document.createElement('span');
          try {
            katex.render(part.content, span, {
              displayMode: part.display,
              throwOnError: false,
            });
            containerRef.current?.appendChild(span);
          } catch (e) {
            span.textContent = part.display ? `$$${part.content}$$` : `$${part.content}$`;
            containerRef.current?.appendChild(span);
          }
        }
      });
    };

    renderMath();
  }, [content]);

  return <div ref={containerRef} className={`math-content ${className}`} />;
}
