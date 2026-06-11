import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { getShow } from "@/lib/api";
import { UI_STRINGS } from "@/lib/i18n";
import type { Lang, RecCard, ShowDetails } from "@/lib/types";

const TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342";

interface ShowDetailsModalProps {
  show: RecCard | null;
  lang: Lang;
  onClose: () => void;
}

export function ShowDetailsModal({ show, lang, onClose }: ShowDetailsModalProps) {
  if (!show) return null;

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      {/* Re-mount when the show or language changes so per-show fetch state resets cleanly. */}
      <ShowDetailsContent key={`${show.title}:${lang}`} show={show} lang={lang} onClose={onClose} />
    </Dialog>
  );
}

interface ShowDetailsContentProps {
  show: RecCard;
  lang: Lang;
  onClose: () => void;
}

function ShowDetailsContent({ show, lang, onClose }: ShowDetailsContentProps) {
  const strings = UI_STRINGS[lang];
  const [details, setDetails] = useState<ShowDetails | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    getShow(show.title, lang)
      .then((result) => {
        if (!cancelled) setDetails(result);
      })
      .catch(() => {
        // TMDB lookup failed or had no match - keep showing catalog data only.
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [show, lang]);

  return (
    <DialogContent dir={lang === "he" ? "rtl" : "ltr"} className="max-h-[85vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>{show.title}</DialogTitle>
        <DialogDescription>
          {show.genres} · {show.decade_str}
          {show.num_seasons != null ? ` · ${strings.seasonsLabel}: ${show.num_seasons}` : ""}
        </DialogDescription>
      </DialogHeader>

      <div className="space-y-3 text-sm">
        {show.poster_path && (
          <img
            src={`${TMDB_POSTER_BASE}${show.poster_path}`}
            alt={show.title}
            className="mx-auto h-64 rounded-md object-cover"
          />
        )}

        <Badge variant="secondary">
          {strings.ratingLabel}: {show.rating.toFixed(1)}
        </Badge>

        <p>{show.overview}</p>

        {show.explanation && (
          <p className="rounded-md bg-muted p-2 italic text-muted-foreground">
            {show.explanation}
          </p>
        )}

        {loading && <p className="text-muted-foreground">{strings.detailsLoading}</p>}

        {details?.trailer_url && (
          <a
            href={details.trailer_url}
            target="_blank"
            rel="noreferrer"
            className="text-primary underline"
          >
            {strings.trailerLabel}
          </a>
        )}

        {details && details.cast.length > 0 && (
          <p>
            <span className="font-semibold">{strings.castLabel}: </span>
            {details.cast.map((member, index) => (
              <span key={member}>
                {index > 0 ? ", " : ""}
                {member}
              </span>
            ))}
          </p>
        )}

        {details && details.watch_providers.length > 0 && (
          <p>
            <span className="font-semibold">{strings.watchProvidersLabel}: </span>
            {details.watch_providers.map((provider, index) => (
              <span key={provider}>
                {index > 0 ? ", " : ""}
                {provider}
              </span>
            ))}
          </p>
        )}
      </div>

      <DialogFooter>
        <Button variant="secondary" onClick={onClose}>
          {strings.closeModal}
        </Button>
      </DialogFooter>
    </DialogContent>
  );
}
