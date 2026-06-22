import { useState } from "react";
import { Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { SeedPickMessage } from "@/lib/chatReducer";
import { UI_STRINGS } from "@/lib/i18n";
import type { Lang, RecCard as RecCardData } from "@/lib/types";

const TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342";

interface SeedPickerProps {
  message: SeedPickMessage;
  lang: Lang;
  onToggle: (title: string) => void;
  onConfirm: () => void;
  onSkip: () => void;
}

function SeedCard({
  show,
  lang,
  selected,
  disabled,
  onToggle,
}: {
  show: RecCardData;
  lang: Lang;
  selected: boolean;
  disabled: boolean;
  onToggle: (title: string) => void;
}) {
  const strings = UI_STRINGS[lang];
  const [imgError, setImgError] = useState(false);
  return (
    <Card
      role="button"
      aria-pressed={selected}
      tabIndex={0}
      onClick={() => onToggle(show.title)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onToggle(show.title);
        }
      }}
      className={`relative w-32 shrink-0 overflow-hidden transition hover:-translate-y-0.5 ${
        selected ? "border-primary ring-2 ring-primary" : "hover:border-primary/50"
      } ${disabled && !selected ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}
    >
      {selected && (
        <div className="absolute right-1 top-1 z-10 rounded-full bg-primary p-0.5 text-primary-foreground">
          <Check className="h-3.5 w-3.5" />
        </div>
      )}
      {show.poster_path && !imgError ? (
        <img
          src={`${TMDB_POSTER_BASE}${show.poster_path}`}
          alt={show.title}
          className="h-44 w-full object-cover"
          onError={() => setImgError(true)}
        />
      ) : (
        <div className="flex h-44 w-full items-center justify-center bg-muted p-2 text-center text-xs text-muted-foreground">
          {show.title}
        </div>
      )}
      <CardContent className="space-y-1 p-2" dir={lang === "he" ? "rtl" : "ltr"}>
        <p className="truncate text-xs font-semibold">{show.title}</p>
        <Badge variant="outline" className="border-primary/40 bg-primary/10 text-primary">
          {strings.ratingLabel}: {show.rating.toFixed(1)}
        </Badge>
      </CardContent>
    </Card>
  );
}

export function SeedPicker({ message, lang, onToggle, onConfirm, onSkip }: SeedPickerProps) {
  const strings = UI_STRINGS[lang];
  const dir = lang === "he" ? "rtl" : "ltr";
  const selected = new Set(message.selectedTitles);
  const atCap = message.selectedTitles.length >= 3;

  return (
    <div dir={dir} className="flex w-full flex-col gap-2">
      <div
        className={`rounded-2xl border border-primary/20 bg-card px-3 py-2 text-sm ${
          lang === "he" ? "self-end" : "self-start"
        }`}
      >
        {strings.seedPrompt}
      </div>

      {message.cards === null ? (
        <div className="px-3 text-sm text-muted-foreground">{strings.seedLoading}</div>
      ) : (
        <div dir="ltr" className={`flex w-full ${lang === "he" ? "justify-end" : "justify-start"}`}>
          <div className="flex max-w-full flex-wrap gap-2">
            {message.cards.map((show) => (
              <SeedCard
                key={show.title}
                show={show}
                lang={lang}
                selected={selected.has(show.title)}
                disabled={message.done || (atCap && !selected.has(show.title))}
                onToggle={message.done ? () => {} : onToggle}
              />
            ))}
          </div>
        </div>
      )}

      {message.cards !== null && !message.done && (
        <div className={`flex gap-2 ${lang === "he" ? "justify-end" : "justify-start"}`}>
          <Button variant="outline" size="sm" className="rounded-full" onClick={onSkip}>
            {strings.seedSkip}
          </Button>
          <Button
            size="sm"
            className="rounded-full"
            disabled={message.selectedTitles.length === 0}
            onClick={onConfirm}
          >
            {strings.seedContinue}
          </Button>
        </div>
      )}
    </div>
  );
}
