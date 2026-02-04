document.addEventListener("click", (e) => {
    const btn = e.target.closest(".quiz__choice");
    if (!btn) return;
  
    const quiz = btn.closest(".quiz");
    const correct = (quiz.dataset.answer || "").trim().toLowerCase();
    const chosen = (btn.dataset.choice || "").trim().toLowerCase();
  
    if (quiz.dataset.answered === "true") return;
    quiz.dataset.answered = "true";
  
    const feedback = quiz.querySelector(".quiz__feedback");
    const explanation = quiz.querySelector(".quiz__explanation");
  
    const isCorrect = chosen === correct;
  
    quiz.querySelectorAll(".quiz__choice").forEach(b => {
      b.disabled = true;
      b.setAttribute("aria-disabled", "true");
      if (b.dataset.choice === correct) b.classList.add("is-correct");
      if (b === btn && !isCorrect) b.classList.add("is-wrong");
    });
  
    feedback.textContent = isCorrect ? "✅ Correct." : "❌ Not quite.";
    explanation.style.display = "block";
  });
  