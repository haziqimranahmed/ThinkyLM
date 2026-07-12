"""
ThinkyLM — Sample Data Preparation
=====================================
Creates a small, legally clean sample dataset from original text.
This is demonstration data only — clearly labelled as such.

Usage:
    python data_pipeline/prepare_sample_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

SAMPLE_TEXTS = [
    # Original text, written specifically for ThinkyLM (MIT-licensed)
    """Philosophy is the systematic inquiry into questions that matter but resist easy answers.
It differs from science not in rigour but in subject matter: philosophy addresses questions
about meaning, value, knowledge, and the foundations of reasoning itself.

A good argument has premises that are true and an inferential step that is valid. A sound
argument has both. Many arguments in everyday discourse are valid but unsound, or neither.
Distinguishing these cases is a useful skill.

The person who says I cannot be wrong is almost certainly wrong about something. Certainty
is expensive and should be purchased with commensurate evidence. Where evidence is thin,
confidence should be proportionately restrained.

Assumptions are the load-bearing walls of arguments. When an assumption is hidden, the
argument stands up only because no one has thought to look for the wall. Critical thinking
begins by making walls visible.

The appeal to popularity is not an argument. That many people believe a thing is evidence
about what many people believe, not about whether the thing is true. History is littered with
very popular mistakes.

To change one's mind in response to evidence is not inconsistency. It is the correct
response. Refusing to change one's mind in response to evidence, on the other hand, is
a failure of rationality, not a display of integrity.
""",
    """Epistemology is the branch of philosophy concerned with knowledge: what it is, how it
is acquired, and what its limits are. The Socratic tradition holds that the beginning of
wisdom is recognising the extent of one's ignorance.

A belief is justified if it is based on good reasons. A belief is true if it corresponds
to how things actually are. Knowledge has traditionally been defined as justified true belief,
though Gettier counterexamples have shown that this definition requires refinement.

Scepticism in its global form holds that we cannot know anything with certainty. In its
local form, it targets specific domains: we cannot know the external world, or the past,
or other minds. Local scepticism is more interesting because it forces one to ask what
knowledge in those domains would actually require.

Inference to the best explanation is a common pattern of reasoning: we accept the hypothesis
that best explains the available evidence. This is not deductively valid — the best available
explanation could still be false — but it is often our most reasonable path forward.

Confirmation bias is the tendency to notice and weight evidence that supports one's existing
beliefs while discounting contrary evidence. It is one of the most reliably demonstrated
failures of human reasoning and is not easily corrected by knowing that it exists.
""",
    """Ethics is the philosophical study of how we ought to live and what we ought to do. It
encompasses moral theory, applied ethics, and metaethics. The three major traditions are
consequentialism, deontology, and virtue ethics.

Consequentialism evaluates actions by their outcomes. An action is right if it produces
better consequences than the alternatives. The classic difficulty is that this can justify
horrifying actions if the numbers are right. The defender of consequentialism must either
accept this consequence, bite the bullet, or argue that properly calculated consequences
never actually justify what common sense forbids.

Deontology evaluates actions by their conformity to rules or duties. Kant's categorical
imperative holds that one should act only on maxims that could be universalised without
contradiction. The difficulty is specifying the relevant description under which an action
falls and what to do when duties conflict.

Virtue ethics focuses on character rather than acts. The question is not what should I do
but what kind of person should I become. The virtuous person perceives what is salient in a
situation and responds appropriately, having cultivated the relevant dispositions through
practice. The difficulty is specifying which virtues are central and how to act when they
conflict.

Moral intuitions are data points. When a carefully constructed argument yields a conclusion
that strikes virtually everyone as monstrous, this is evidence that one of the premises is
false, not that we must accept the monstrous conclusion. Philosophy is not a machine for
overriding intuitions; it is a tool for examining and sometimes revising them.
""",
    """Logic is the study of valid inference. A valid argument is one in which the truth of
the premises guarantees the truth of the conclusion. A sound argument is valid and has
true premises. Validity is a matter of form; soundness is a matter of form and content.

The modus ponens is the most basic valid argument form: if P then Q; P; therefore Q. The
modus tollens is equally basic: if P then Q; not Q; therefore not P. Both are deductively
valid. Arguments that resemble these forms but are not actually instances of them are
fallacious — affirming the consequent and denying the antecedent are the canonical cases.

The ad hominem fallacy attacks the person making the argument rather than the argument
itself. It is fallacious because the truth or validity of an argument does not depend on
the character of the person who advances it. Note, however, that attacking credentials
is relevant when the issue is whether to trust a testimony, not whether an argument is valid.

The false dichotomy presents two options as exhaustive when others exist. Either you are
with us or against us is rarely literally true. Identifying the false dichotomy is often
the key move in dissolving an apparent dilemma.

Circular reasoning assumes what it seeks to prove. The conclusion appears as one of the
premises, disguised. Detection requires making the argument's structure explicit and asking
whether any premise would not be accepted by someone who doubted the conclusion.
""",
    """The philosophy of mind addresses questions about the nature of mental states, their
relationship to physical states, and whether machines could be minds.

The mind-body problem asks how mental states and physical states are related. Dualism holds
that mind and body are distinct substances. Physicalism holds that mental states are physical
states, though there are many varieties: identity theory, functionalism, eliminativism. Each
has costs and benefits.

The hard problem of consciousness, as articulated by Chalmers, is not about explaining
how the brain processes information — that is the easy problem — but about explaining why
information processing is accompanied by subjective experience at all. Why is there something
it is like to be me?

Functionalism holds that mental states are defined by their functional role — what they are
caused by and what they cause — rather than by their physical substrate. If something performs
the function of pain, it is in pain, regardless of whether it is made of neurons or silicon.
Opponents argue that this misses the felt quality of experience.

The Turing test proposes that if a machine can converse in a way indistinguishable from a
human, we should credit it with intelligence. Searle's Chinese room argument holds that
passing the Turing test is insufficient for understanding: a system could manipulate symbols
according to rules and produce correct outputs without understanding anything at all.

ThinkyLM is not claimed to understand. It is an architecture trained on symbols. Whether
symbol manipulation can give rise to understanding is precisely the question the Chinese
room raises, and it has not been resolved.
""",
]


def prepare_sample_data(output_dir: Path) -> None:
    """Write sample text files to the sample data directory.

    Args:
        output_dir: Destination directory for sample text files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("ThinkyLM — Sample Data Preparation")
    print("=" * 55)
    print("NOTE: All text below is original content written for")
    print("ThinkyLM under the MIT licence. It is demonstration")
    print("data, NOT a full training corpus.")
    print()

    topics = [
        "philosophy_intro",
        "epistemology",
        "ethics",
        "logic_fallacies",
        "philosophy_of_mind",
    ]

    total_chars = 0
    total_words = 0

    for i, (topic, text) in enumerate(zip(topics, SAMPLE_TEXTS)):
        file_path = output_dir / f"{topic}.txt"
        file_path.write_text(text.strip(), encoding="utf-8")
        chars = len(text)
        words = len(text.split())
        total_chars += chars
        total_words += words
        print(f"  [{i+1}] {file_path.name}: {chars:,} chars, {words:,} words")

    # Also copy the tokenizer sample
    tok_sample = Path("tokenizer/sample_text.txt")
    if tok_sample.exists():
        dest = output_dir / "sample_from_tokenizer.txt"
        shutil.copy(tok_sample, dest)
        text = dest.read_text(encoding="utf-8")
        total_chars += len(text)
        total_words += len(text.split())
        print(f"  [+] {dest.name}: {len(text):,} chars")

    print()
    print(f"Total: {total_chars:,} chars | {total_words:,} words")
    print(f"Output: {output_dir.resolve()}")
    print()
    print("Licence: MIT (original ThinkyLM content)")
    print("Status : DEMONSTRATION DATA — not a production corpus")


def main() -> None:
    output_dir = Path("data/sample")
    prepare_sample_data(output_dir)


if __name__ == "__main__":
    main()
