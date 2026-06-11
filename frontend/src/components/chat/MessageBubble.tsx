import { cn } from "@/lib/utils";
import type { TextMessage } from "@/lib/chatReducer";
import type { Lang } from "@/lib/types";

interface MessageBubbleProps {
  message: TextMessage;
  lang: Lang;
}

export function MessageBubble({ message, lang }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        dir={lang === "he" ? "rtl" : "ltr"}
        className={cn(
          "max-w-[80%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "border border-border bg-card text-card-foreground"
        )}
      >
        {message.content}
      </div>
    </div>
  );
}
