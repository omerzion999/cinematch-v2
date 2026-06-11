import { ChatWindow } from "@/components/chat/ChatWindow";

export default function App() {
  return (
    <>
      <div
        aria-hidden
        className="fixed inset-0 -z-10 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage:
            'linear-gradient(180deg, hsl(var(--background) / 0.45), hsl(var(--background) / 0.7)), url("/background.png")',
        }}
      />
      <ChatWindow />
    </>
  );
}
