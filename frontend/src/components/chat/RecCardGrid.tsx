import { RecCard } from "./RecCard";
import type { RecommendationsMessage } from "@/lib/chatReducer";
import type { Lang } from "@/lib/types";

interface RecCardGridProps {
  message: RecommendationsMessage;
  lang: Lang;
  onSelectShow: (title: string) => void;
}

export function RecCardGrid({ message, lang, onSelectShow }: RecCardGridProps) {
  return (
    <div dir="ltr" className={`flex w-full ${lang === "he" ? "justify-end" : "justify-start"}`}>
      <div className="flex max-w-full flex-wrap gap-3">
        {message.cards.map((show) => (
          <RecCard key={show.title} show={show} lang={lang} onClick={onSelectShow} />
        ))}
      </div>
    </div>
  );
}
