import ChatInput from '../ChatInput';

export default function ChatInputExample() {
  return (
    <div className="h-32">
      <ChatInput
        onSendMessage={(msg) => console.log('Message sent:', msg)}
      />
    </div>
  );
}
