import FeedbackControls from '../FeedbackControls';

export default function FeedbackControlsExample() {
  return (
    <div className="p-6 max-w-md">
      <FeedbackControls
        messageId="msg-1"
        onSubmitFeedback={(id, rating, comment) => {
          console.log('Feedback submitted:', { id, rating, comment });
        }}
      />
    </div>
  );
}
