"""The sentence rules, which are pure text and need no Home Assistant.

Every case here is a trap that cost something real:

* a stored sentence carrying `, ; ? !` matches nothing — hassil treats them as separators and
  Home Assistant strips them from the *input* but not from the stored sentence, so the phrase
  is reported as learned and then silently never works;
* a phrase made only of follow-up words ("when?") means whatever the previous turn said, so
  learning it pins it to what it happened to mean once;
* an intent file named after the note that produced it says nothing about what is inside.

The tests that matter for matching itself (does hassil accept this?) live in
`test_hassil_rules.py`, which calls hassil for real.
"""

from __future__ import annotations

import unittest

from support import HassCase  # noqa: F401  (kept for symmetry; these need no hass)

from custom_components.irabelle_home_learning.autolearn import reply_problems
from custom_components.irabelle_home_learning.learner import (
    names_nothing,
    tidy_punctuation,
    normalize_sentence,
    sentence_for_storage,
    slugify,
)


class SentenceStorage(unittest.TestCase):
    def test_sentence_punctuation_is_removed(self):
        # hassil treats these as separators; the stored sentence must not carry one.
        self.assertEqual(
            sentence_for_storage("when was my phone last seen?"),
            "when was my phone last seen",
        )
        self.assertEqual(
            sentence_for_storage("when was my phone last seen, please"),
            "when was my phone last seen please",
        )
        self.assertEqual(sentence_for_storage("is it cold!"), "is it cold")

    def test_apostrophes_and_hyphens_survive(self):
        # They occur inside words and match fine, so stripping them would lose wording.
        self.assertEqual(sentence_for_storage("where's my phone"), "where's my phone")
        self.assertEqual(sentence_for_storage("tv-guide please"), "tv-guide please")

    def test_trailing_period_goes_but_a_decimal_point_stays(self):
        self.assertEqual(sentence_for_storage("it is 16."), "it is 16")
        self.assertEqual(sentence_for_storage("it is 16.5 degrees"), "it is 16.5 degrees")

    def test_runs_of_whitespace_collapse(self):
        self.assertEqual(sentence_for_storage("what   time\n is it"), "what time is it")


class SentenceNormalisation(unittest.TestCase):
    """The form used to compare two sentences for collision, not the form that is stored."""

    def test_case_and_punctuation_fold(self):
        self.assertEqual(normalize_sentence("Who I am?"), "who i am")

    def test_optional_groups_and_slots_are_removed(self):
        # `[{speaker} ]do you know who I am` and a learned "do you know who I am" are the
        # same request; whichever wins would otherwise be arbitrary.
        self.assertEqual(
            normalize_sentence("[{speaker} ]do you know who I am"), "do you know who i am"
        )
        self.assertEqual(normalize_sentence("what's the weather {day}"), "what s the weather")

    def test_different_sentences_stay_different(self):
        self.assertNotEqual(
            normalize_sentence("is the bedroom tv on"), normalize_sentence("is the tv on")
        )


class FollowUpWords(unittest.TestCase):
    def test_a_bare_follow_up_names_nothing(self):
        self.assertTrue(names_nothing("when"))
        self.assertTrue(names_nothing("and then?"))
        self.assertTrue(names_nothing("what about that"))

    def test_a_real_request_is_not_a_follow_up(self):
        self.assertFalse(names_nothing("who am i"))
        self.assertFalse(names_nothing("what time is it"))
        self.assertFalse(names_nothing("turn off the office light"))

    def test_empty_is_not_a_request(self):
        self.assertFalse(names_nothing(""))


class Slugs(unittest.TestCase):
    def test_intent_names_become_file_safe(self):
        self.assertEqual(slugify("WeatherForecast"), "weatherforecast")
        self.assertEqual(slugify("HassTurnOn"), "hassturnon")

    def test_bounded_and_never_empty(self):
        self.assertLessEqual(len(slugify("x" * 200)), 60)
        self.assertEqual(slugify("!!!"), "learned")


class ReplyWording(unittest.TestCase):
    """`reply_problems` is the mechanical half of "does this read like a sentence"."""

    def test_empty(self):
        self.assertEqual(reply_problems(""), ["it is empty"])

    def test_missing_capital_or_full_stop(self):
        problems = reply_problems("your phone was last seen in the Bedroom")
        self.assertIn("it does not start with a capital letter", problems)
        self.assertIn("it does not end with a full stop", problems)

    def test_a_value_that_did_not_render(self):
        # A doubled space is what an empty `{{ value }}` leaves behind.
        self.assertIn(
            "it has a doubled space, so a value did not render",
            reply_problems("It is  degrees outside."),
        )

    def test_an_unrendered_template(self):
        self.assertIn(
            "a template expression was not rendered",
            reply_problems("It is {{ temperature }} outside."),
        )

    def test_a_thing_said_twice(self):
        self.assertIn(
            "it says 'last seen' twice",
            reply_problems("Your phone was last seen in the Bedroom, last seen 4 minutes ago."),
        )

    def test_two_half_sentences_glued_together(self):
        # What a model's rewrite of an existing answer produced: each half was fine, the join
        # was not, and nothing else here looks at punctuation.
        problems = reply_problems(
            "Showers clearing in the evening., the high will be 19 and the low 10 degrees."
        )
        self.assertTrue(
            any("two punctuation marks in a row" in problem for problem in problems), problems
        )

    def test_a_comma_before_a_full_stop_is_caught(self):
        problems = reply_problems("A little rain is expected,. And the UV index is 1.7.")
        self.assertTrue(any("two punctuation marks in a row" in p for p in problems), problems)

    def test_a_value_that_did_not_resolve_is_not_spoken(self):
        # Measured: on a turn with no chat user and no recognised voice, the speaker sensor
        # reads `unknown`, and the replies that print it unguarded answered "unknown, your
        # Garmin watch was last seen in the Office". A missing sensor reads `unavailable`.
        problems = reply_problems("unknown, your Garmin watch was last seen in the Office.")
        self.assertTrue(any("did not resolve" in problem for problem in problems), problems)
        for text in (
            "It is unavailable.",
            "The office light is unknown.",
            "The living room light is unknown and the Hue Iris is unknown.",
        ):
            with self.subTest(text=text):
                self.assertTrue(
                    any("did not resolve" in problem for problem in reply_problems(text)), text
                )

    def test_a_reply_that_explains_the_missing_value_is_fine(self):
        # The word only counts when it stands in for the answer. Saying what cannot be done is
        # the *good* version of the same reply, and flagging it would be noise.
        self.assertEqual(
            reply_problems(
                "The office light's state is unknown so I can't tell when it was switched on."
            ),
            [],
        )

    def test_a_guard_that_mentions_those_words_is_fine(self):
        # The word only matters in the output: a template may test for it all it likes.
        self.assertEqual(reply_problems("Your watch is in the Office."), [])

    def test_a_number_and_its_unit_have_to_agree(self):
        # Measured: "your iPhone Alex was last seen in the Bedroom, 1 hours ago".
        self.assertTrue(
            any(
                "do not agree" in problem
                for problem in reply_problems("It was last seen 1 hours ago.")
            )
        )
        self.assertEqual(reply_problems("It was last seen 2 hours ago."), [])

    def test_an_ellipsis_is_a_deliberate_pause(self):
        self.assertEqual(
            reply_problems("It is not reporting... I could not tell you."),
            [],
        )

    def test_a_joined_sentence_that_kept_its_capital(self):
        # The text itself proves the word is not a proper noun: it writes it lowercase too.
        problems = reply_problems(
            "Today it is Isolated showers, becoming widespread, Showers clearing in the evening."
        )
        self.assertTrue(any("in the middle of a sentence" in p for p in problems), problems)

    def test_a_proper_noun_is_not_a_join(self):
        # "Sam" appears once, so the text gives no reason to think it is not a name.
        self.assertEqual(
            reply_problems("It will rain this morning, Sam will be pleased."), []
        )

    def test_a_word_written_lowercase_after_a_comma_is_fine(self):
        self.assertEqual(
            reply_problems("It will rain this morning, clearing by the evening."), []
        )

    def test_a_good_sentence_has_no_problems(self):
        self.assertEqual(reply_problems("Your phone was last seen in the Bedroom."), [])
        self.assertEqual(
            reply_problems("Wear a jacket, closed shoes and bring an umbrella."), []
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TidyPunctuation(unittest.TestCase):
    """The one punctuation mistake that is a typo rather than a wording choice.

    This is what a model's rewrite of an existing answer produces, and the polish pass (asked
    of the same model) answered "no rewrite offered" often enough that the component fixes it
    itself on write.
    """

    def test_a_full_stop_before_a_joining_comma_goes(self):
        self.assertEqual(
            tidy_punctuation("in the afternoon., the high will be 19"),
            "in the afternoon, the high will be 19",
        )

    def test_it_keeps_the_full_stop_when_a_new_sentence_starts(self):
        self.assertEqual(
            tidy_punctuation("it is raining., The high will be 19"),
            "it is raining. The high will be 19",
        )

    def test_a_repeated_mark_collapses(self):
        self.assertEqual(tidy_punctuation("really!!"), "really!")
        self.assertEqual(tidy_punctuation("wait??"), "wait?")
        self.assertEqual(tidy_punctuation("done.."), "done.")

    def test_an_ellipsis_is_left_alone(self):
        self.assertEqual(tidy_punctuation("it is not reporting..."), "it is not reporting...")

    def test_ordinary_text_is_untouched(self):
        for text in ("It is 3.5 degrees, with 60% humidity.", "Today it is cloudy.", "3 p.m."):
            self.assertEqual(tidy_punctuation(text), text)
