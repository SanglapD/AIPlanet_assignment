import ChatHeader from '../ChatHeader';

export default function ChatHeaderExample() {
  return (
    <ChatHeader 
      onClearChat={() => console.log('Clear chat clicked')} 
    />
  );
}
