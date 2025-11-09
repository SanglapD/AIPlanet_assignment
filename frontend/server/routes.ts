import type { Express } from "express";
import { createServer, type Server } from "http";

interface ChatRequest {
  question: string;
}

interface FeedbackRequest {
  message_id: string;
  rating: "positive" | "negative";
  comment?: string;
}

const mockMathSolutions: Record<string, { answer: string; route: string }> = {
  "derivative": {
    answer: `To find the derivative of $\\sin(x)$, we use the standard derivative rule.

**Solution:**

The derivative of $\\sin(x)$ with respect to $x$ is:

$$\\frac{d}{dx}[\\sin(x)] = \\cos(x)$$

**Explanation:**

This is one of the fundamental trigonometric derivatives. The rate of change of the sine function at any point $x$ is given by the cosine of that same point.

**Example:**
If $f(x) = \\sin(x)$, then $f'(x) = \\cos(x)$

At $x = 0$: $f'(0) = \\cos(0) = 1$
At $x = \\frac{\\pi}{2}$: $f'(\\frac{\\pi}{2}) = \\cos(\\frac{\\pi}{2}) = 0$`,
    route: "knowledge_base"
  },
  "quadratic": {
    answer: `Let's solve the quadratic equation $x^2 - 5x + 6 = 0$.

**Step 1: Identify coefficients**
- $a = 1$
- $b = -5$
- $c = 6$

**Step 2: Use the quadratic formula**

$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

**Step 3: Calculate the discriminant**

$$\\Delta = b^2 - 4ac = (-5)^2 - 4(1)(6) = 25 - 24 = 1$$

**Step 4: Solve for x**

$$x = \\frac{5 \\pm \\sqrt{1}}{2} = \\frac{5 \\pm 1}{2}$$

**Solutions:**
- $x_1 = \\frac{5 + 1}{2} = 3$
- $x_2 = \\frac{5 - 1}{2} = 2$

**Verification:**
- $(x-2)(x-3) = x^2 - 5x + 6$ ✓

Therefore, $x = 2$ or $x = 3$.`,
    route: "knowledge_base"
  },
  "pythagorean": {
    answer: `The **Pythagorean Theorem** is a fundamental principle in geometry.

**Statement:**

In a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of squares of the lengths of the other two sides.

**Formula:**

$$a^2 + b^2 = c^2$$

where:
- $c$ is the length of the hypotenuse
- $a$ and $b$ are the lengths of the other two sides

**Example:**

Consider a right triangle with sides $a = 3$ and $b = 4$.

$$c^2 = 3^2 + 4^2 = 9 + 16 = 25$$

$$c = \\sqrt{25} = 5$$

**Applications:**
1. Distance calculations in coordinate geometry
2. Navigation and surveying
3. Construction and architecture
4. Computer graphics and game development

This theorem only applies to right triangles (triangles with a 90° angle).`,
    route: "knowledge_base"
  },
  "integrate": {
    answer: `Let's integrate $\\int x^2 \\, dx$.

**Step 1: Apply the power rule for integration**

The power rule states: $\\int x^n \\, dx = \\frac{x^{n+1}}{n+1} + C$ (where $n \\neq -1$)

**Step 2: Apply to our integral**

For $\\int x^2 \\, dx$, we have $n = 2$:

$$\\int x^2 \\, dx = \\frac{x^{2+1}}{2+1} + C = \\frac{x^3}{3} + C$$

**Final Answer:**

$$\\int x^2 \\, dx = \\frac{x^3}{3} + C$$

where $C$ is the constant of integration.

**Verification:**

We can verify by taking the derivative:
$$\\frac{d}{dx}\\left[\\frac{x^3}{3} + C\\right] = \\frac{3x^2}{3} = x^2$$ ✓`,
    route: "knowledge_base"
  }
};

function getMockResponse(question: string): { answer: string; route: string } {
  const lowerQuestion = question.toLowerCase();
  
  if (lowerQuestion.includes("derivative") || lowerQuestion.includes("sin")) {
    return mockMathSolutions.derivative;
  } else if (lowerQuestion.includes("x²") || lowerQuestion.includes("x^2") || lowerQuestion.includes("5x")) {
    return mockMathSolutions.quadratic;
  } else if (lowerQuestion.includes("pythagorean")) {
    return mockMathSolutions.pythagorean;
  } else if (lowerQuestion.includes("integrate") || lowerQuestion.includes("∫")) {
    return mockMathSolutions.integrate;
  }
  
  if (lowerQuestion.length > 100 || lowerQuestion.includes("latest") || lowerQuestion.includes("recent")) {
    return {
      answer: `Based on web search results, here's what I found:

${question}

**Analysis:**

This appears to be a question that requires current information or isn't in my knowledge base. In a production environment, this would trigger a web search using the Model Context Protocol (MCP) with Tavily integration.

**Approach:**

1. The system would query the web for relevant information
2. Extract key mathematical concepts
3. Generate a step-by-step explanation
4. Validate the solution

For demonstration purposes, this response shows how the routing system differentiates between knowledge base queries and those requiring web search.`,
      route: "web_search"
    };
  }
  
  return {
    answer: `I'll help you with: "${question}"

**Understanding the Question:**

Let me break this down step by step.

**Step 1: Identify the Problem Type**

This appears to be a mathematical question that requires analysis.

**Step 2: Apply Relevant Principles**

We'll use fundamental mathematical concepts to solve this.

**Step 3: Solution**

Let me work through this systematically:
- First, we identify the given information
- Then, we apply appropriate formulas or theorems
- Finally, we verify our result

**Note:** This is a demonstration response. In production, the system would:
- Check the FAISS vector database for similar problems
- Route to web search if not found in knowledge base
- Apply guardrails to ensure educational content only
- Generate detailed step-by-step solutions

The actual Math Tutor AI uses LangChain, Ollama LLM, and MCP for comprehensive mathematical problem-solving.`,
    route: "knowledge_base"
  };
}

export async function registerRoutes(app: Express): Promise<Server> {
  
  // Proxy chat requests to the Python backend on localhost:8000.  This endpoint
  // forwards the question to the FastAPI /chat route and streams the returned
  // answer back to the client as Server‑Sent Events (SSE).
  app.post("/api/chat", async (req, res) => {
    const { question } = req.body as ChatRequest;

    // Validate input
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Question is required" });
    }

    try {
      // Forward the question to the Python /chat endpoint.  Use global fetch;
      // the FastAPI server must be running on port 8000.  The response
      // includes an answer string and a record_id that uniquely identifies
      // this conversation.
      const apiRes = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!apiRes.ok) {
        const text = await apiRes.text();
        return res.status(apiRes.status).json({ error: text || "Unknown error" });
      }

      const data = await apiRes.json() as { answer: string; record_id: string; tool_used?: string };
      const { answer, record_id, tool_used } = data;

      // Stream the answer back to the client word by word via SSE.  Use the
      // record_id from the backend as the message_id.  The backend does not
      // provide routing information, so this field is omitted.
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const words = answer.split(" ");
      let sentContent = "";

      for (let i = 0; i < words.length; i++) {
        const word = words[i] + (i < words.length - 1 ? " " : "");
        sentContent += word;

        res.write(`data: ${JSON.stringify({
          content: word,
          message_id: i === 0 ? record_id : undefined,
          tool_used: i === 0 ? tool_used : undefined,
        })}\n\n`);

        // Introduce a slight delay between words to simulate streaming
        await new Promise((resolve) => setTimeout(resolve, 30 + Math.random() * 40));
      }

      res.write(`data: ${JSON.stringify({
        done: true,
        message_id: record_id,
        answer: sentContent,
        tool_used: tool_used,
      })}\n\n`);

      res.end();
    } catch (err: any) {
      console.error(err);
      return res.status(500).json({ error: "Failed to call backend" });
    }
  });

  // Proxy feedback requests to the Python backend.  The incoming request
  // contains a message_id, a textual rating ("positive"/"negative"), and an
  // optional comment.  These values are mapped to the fields expected by the
  // Python /feedback API.
  app.post("/api/feedback", async (req, res) => {
    const { message_id, rating, comment } = req.body as FeedbackRequest;

    if (!message_id || !rating) {
      return res.status(400).json({ error: "message_id and rating are required" });
    }

    // Translate the string rating to a numeric value expected by the backend
    const ratingMap: Record<string, number> = {
      positive: 5,
      negative: 1,
    };
    const numericRating = ratingMap[rating] ?? 3;

    try {
      const apiRes = await fetch("http://localhost:8000/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          record_id: message_id,
          rating: numericRating,
          comments: comment,
        }),
      });

      if (!apiRes.ok) {
        const text = await apiRes.text();
        return res.status(apiRes.status).json({ error: text || "Unknown error" });
      }

      const data = await apiRes.json() as { success: boolean; updated_answer: string | null };
      return res.json(data);
    } catch (err: any) {
      console.error(err);
      return res.status(500).json({ error: "Failed to call backend" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
