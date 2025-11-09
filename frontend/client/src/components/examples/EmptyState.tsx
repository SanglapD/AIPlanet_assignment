import EmptyState from '../EmptyState';

export default function EmptyStateExample() {
  return (
    <div className="h-[500px]">
      <EmptyState
        onExampleClick={(question) => console.log('Example clicked:', question)}
      />
    </div>
  );
}
