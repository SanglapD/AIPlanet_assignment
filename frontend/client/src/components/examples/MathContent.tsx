import MathContent from '../MathContent';

export default function MathContentExample() {
  const sampleContent = `Let's solve the quadratic equation $ax^2 + bx + c = 0$.

The quadratic formula is:

$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

For example, if $a=1$, $b=-3$, and $c=2$, we get:

$$x = \\frac{3 \\pm \\sqrt{9 - 8}}{2} = \\frac{3 \\pm 1}{2}$$

So $x = 2$ or $x = 1$.`;

  return (
    <div className="p-6 max-w-2xl">
      <MathContent content={sampleContent} className="text-base leading-loose" />
    </div>
  );
}
