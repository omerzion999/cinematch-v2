import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatState } from "@/hooks/useChatState";
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

  return (
    <div dir={dir} className="mx-auto flex h-screen max-w-2xl flex-col bg-background">
      <header className="flex items-center justify-between border-b border-border p-3">
        <h1 className="text-lg font-bold">CineMatch AI</h1>
        <Button variant="ghost" size="sm" onClick={() => dispatch({ type: "TOGGLE_LANG" })}>
          {strings.languageToggleLabel}
        </Button>
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
            <div className="flex w-full justify-start">
              <div className="rounded-lg border border-border bg-card px-3 py-2 text-sm text-muted-foreground">
                {strings.processing}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <form
        className="flex items-center gap-2 border-t border-border p-3"
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
        />
        <Button type="submit" disabled={!canSendMessage || inputValue.trim().length === 0}>
          {strings.send}
        </Button>
      </form>

      <ShowDetailsModal
        show={selectedShow}
        lang={state.lang}
        onClose={() => setSelectedShow(null)}
      />
    </div>
  );
}
