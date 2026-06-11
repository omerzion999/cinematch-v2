import { useEffect, useRef, useState } from "react";
import { ArrowRight, ArrowLeft, RotateCcw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatState } from "@/hooks/useChatState";
import { postTranslate } from "@/lib/api";
import { findStaticTranslation, type TextMessage } from "@/lib/chatReducer";
import { UI_STRINGS } from "@/lib/i18n";
import { ONBOARDING_QUESTIONS } from "@/lib/onboarding";
import type { RecCard as RecCardData } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";
import { OnboardingQuestion } from "./OnboardingQuestion";
import { RecCardGrid } from "./RecCardGrid";
import { ShowDetailsModal } from "./ShowDetailsModal";

export function ChatWindow() {
  const { state, dispatch } = useChatState("he");
  const strings = UI_STRINGS[state.lang];
  const [inputValue, setInputValue] = useState("");
  const [selectedShow, setSelectedShow] = useState<RecCardData | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: "smooth" });
  }, [state.messages]);

  const isLoading = state.phase === "loading_recommend" || state.phase === "loading_chat";
  const canSendMessage = state.phase === "chat";
  const dir = state.lang === "he" ? "rtl" : "ltr";

  function handleChoiceSelect(value: string) {
    if (state.phase === "intro") {
      if (value === "start") {
        dispatch({ type: "START_ONBOARDING" });
      } else {
        dispatch({ type: "SKIP_TO_CHAT" });
      }
      return;
    }
    if (state.phase === "onboarding") {
      const question = ONBOARDING_QUESTIONS[state.onboardingStepIndex];
      dispatch({ type: "ANSWER_ONBOARDING_QUESTION", questionId: question.id, value });
    }
  }

  function handleSelectShow(title: string) {
    for (const message of state.messages) {
      if (message.type === "recommendations") {
        const found = message.cards.find((card) => card.title === title);
        if (found) {
          setSelectedShow(found);
          return;
        }
      }
    }
  }

  function handleSend() {
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    dispatch({ type: "SEND_USER_MESSAGE", content: trimmed });
    setInputValue("");
  }

  async function handleToggleLang() {
    const fromLang = state.lang;
    const toLang = fromLang === "he" ? "en" : "he";

    // Free-form assistant text that has no static translation needs an LLM
    // round-trip; collect it before TOGGLE_LANG flips state.lang.
    const dynamicMessages = state.messages.filter(
      (message): message is TextMessage =>
        message.type === "text" &&
        message.role === "assistant" &&
        findStaticTranslation(message.content, fromLang, toLang) === null
    );

    dispatch({ type: "TOGGLE_LANG" });

    if (dynamicMessages.length === 0) return;

    try {
      const { translations } = await postTranslate({
        texts: dynamicMessages.map((message) => message.content),
        target_lang: toLang,
      });
      dispatch({
        type: "APPLY_TRANSLATIONS",
        translations: dynamicMessages.map((message, index) => ({
          id: message.id,
          content: translations[index],
        })),
      });
    } catch {
      // Keep the original-language text if translation fails.
    }
  }

  const SendIcon = dir === "rtl" ? ArrowLeft : ArrowRight;

  return (
    <div className="flex min-h-screen items-center justify-center p-3 sm:p-6">
      <div
        dir={dir}
        className="flex h-[calc(100vh-1.5rem)] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-primary/20 bg-card/85 shadow-2xl shadow-black/50 backdrop-blur-md sm:h-[90vh]"
      >
        <header className="flex items-center justify-between gap-2 border-b border-primary/15 px-4 py-3">
          <h1 className="flex items-center gap-2 text-lg font-bold tracking-wide">
            <Sparkles className="h-4 w-4 text-primary" />
            <span className="bg-gradient-to-r from-primary to-amber-200 bg-clip-text text-transparent">
              CineMatch AI
            </span>
            <Sparkles className="h-4 w-4 text-primary" />
          </h1>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              className="rounded-full border-primary/40 text-primary hover:bg-primary/10 hover:text-primary"
              onClick={() => dispatch({ type: "RESET_CONVERSATION" })}
            >
              <RotateCcw className="h-4 w-4" />
              <span className="sr-only">{strings.newConversation}</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="rounded-full border-primary/40 text-primary hover:bg-primary/10 hover:text-primary"
              onClick={() => void handleToggleLang()}
            >
              {strings.languageToggleLabel}
            </Button>
          </div>
        </header>

        <ScrollArea className="flex-1 p-3">
          <div className="flex flex-col gap-3">
            {state.messages.map((message) => {
              switch (message.type) {
                case "text":
                  return <MessageBubble key={message.id} message={message} lang={state.lang} />;
                case "choice":
                  return (
                    <OnboardingQuestion
                      key={message.id}
                      message={message}
                      lang={state.lang}
                      onSelect={handleChoiceSelect}
                    />
                  );
                case "recommendations":
                  return (
                    <RecCardGrid
                      key={message.id}
                      message={message}
                      lang={state.lang}
                      onSelectShow={handleSelectShow}
                    />
                  );
                default:
                  return null;
              }
            })}
            {isLoading && (
              <div
                dir="ltr"
                className={`flex w-full ${state.lang === "he" ? "justify-end" : "justify-start"}`}
              >
                <div className="rounded-2xl border border-border bg-card px-3 py-2 text-sm text-muted-foreground">
                  {strings.processing}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>

        <form
          className="flex items-center gap-2 border-t border-primary/15 p-3"
          onSubmit={(event) => {
            event.preventDefault();
            handleSend();
          }}
        >
          <Input
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder={strings.inputPlaceholder}
            disabled={!canSendMessage}
            className="rounded-full bg-secondary/60"
          />
          <Button
            type="submit"
            size="icon"
            className="shrink-0 rounded-full"
            disabled={!canSendMessage || inputValue.trim().length === 0}
          >
            <SendIcon className="h-4 w-4" />
            <span className="sr-only">{strings.send}</span>
          </Button>
        </form>

        <ShowDetailsModal
          show={selectedShow}
          lang={state.lang}
          onClose={() => setSelectedShow(null)}
        />
      </div>
    </div>
  );
}
