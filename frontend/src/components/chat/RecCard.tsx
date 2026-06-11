import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { UI_STRINGS } from "@/lib/i18n";
import type { Lang, RecCard as RecCardData } from "@/lib/types";

const TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342";

interface RecCardProps {
  show: RecCardData;
  lang: Lang;
  onClick: (title: string) => void;
}

export function RecCard({ show, lang, onClick }: RecCardProps) {
  const strings = UI_STRINGS[lang];
  const [imgError, setImgError] = useState(false);

  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={() => onClick(show.title)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick(show.title);
        }
      }}
      className="w-40 shrink-0 cursor-pointer overflow-hidden transition hover:-translate-y-0.5 hover:border-primary/50 hover:shadow-lg hover:shadow-primary/10"
    >
      {show.poster_path && !imgError ? (
        <img
          src={`${TMDB_POSTER_BASE}${show.poster_path}`}
          alt={show.title}
          className="h-56 w-full object-cover"
          onError={() => setImgError(true)}
        />
      ) : (
        <div className="flex h-56 w-full items-center justify-center bg-muted p-2 text-center text-sm text-muted-foreground">
          {show.title}
        </div>
      )}
      <CardContent className="space-y-1 p-2" dir={lang === "he" ? "rtl" : "ltr"}>
        <p className="truncate text-sm font-semibold">{show.title}</p>
        <p className="truncate text-xs text-muted-foreground">{show.genres}</p>
        <Badge variant="outline" className="border-primary/40 bg-primary/10 text-primary">
          {strings.ratingLabel}: {show.rating.toFixed(1)}
        </Badge>
      </CardContent>
    </Card>
  );
}
