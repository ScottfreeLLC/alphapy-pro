import { useState, useRef, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Bot } from 'lucide-react';
import { chatWithAlfi } from '../../lib/api';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  const chatMutation = useMutation({
    mutationFn: (message: string) => chatWithAlfi(message),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.reply, timestamp: data.timestamp },
      ]);
    },
    onError: (error: Error) => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${error.message}`, timestamp: new Date().toISOString() },
      ]);
    },
  });

  const handleSend = (message: string) => {
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: message, timestamp: new Date().toISOString() },
    ]);
    chatMutation.mutate(message);
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-blue-600/20 flex items-center justify-center">
            <Bot size={20} className="text-blue-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Chat with Alfi</h2>
            <p className="text-xs text-gray-500">Ask about markets, signals, strategies, or performance</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-20 text-gray-500">
            <Bot size={48} className="mx-auto mb-4 text-gray-600" />
            <p className="text-lg font-medium mb-2">Ask Alfi anything</p>
            <div className="text-sm space-y-1">
              <p>"What signals are active right now?"</p>
              <p>"How is the momentum breakout strategy performing?"</p>
              <p>"What's the current market sentiment for NVDA?"</p>
              <p>"Should I be concerned about any risk limits?"</p>
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <ChatMessage
            key={i}
            role={msg.role}
            content={msg.content}
            timestamp={msg.timestamp}
          />
        ))}
        {chatMutation.isPending && (
          <div className="flex items-center gap-2 text-gray-400 text-sm">
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
            Alfi is thinking...
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-800">
        <ChatInput onSend={handleSend} disabled={chatMutation.isPending} />
      </div>
    </div>
  );
}
