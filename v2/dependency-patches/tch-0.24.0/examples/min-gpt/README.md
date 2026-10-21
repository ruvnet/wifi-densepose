## minGPT-rs

This example implements a character-level language model using
a simplified version of [minGPT](https://github.com/karpathy/minGPT).
All the model details as well as the training loop can be found
in `main.rs`.

Besides the model being different, this example is in line with the LSTM
version [char-rnn](https://github.com/LaurentMazare/tch-rs/tree/master/examples/char-rnn).
The same comments apply:
- For training, the model takes as input a text file and learns to predict the
  next character following a given sequence.
- For generation, the model uses a fixed seed, returns the next character
  distribution from which a character is randomly sampled and this is iterated
  to generate a text.

At the end of each training epoch, some sample text is generated and printed.

Any text file can be used as an input, as long as it's large enough for training.
A typical example would be the
[tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
The training text file should be stored in `data/input.txt`.

Compiling and running the example is done via the following command lines:
```bash
cargo run --example min-gpt
```

Here is an example of generated data when training on the Shakespeare dataset at
the end of the first epoch.

```
LORD:
My nown of none, by God's mother,
No more of this country, nor than I
Who have strengthen lamented to make it stay
there; and yet it will no more but abide.

AUTOLYCUS:
Very true, and but a month old.

DORCAS:
Bless me from marrying a usurer!

AUTOLYCUS:
Here's the midwife's name to't, one Mistress
Overdone's own house, for here be made.

EXETER:
The doubt is that which calls so coldly?

GLOUCESTER:
I grant ye.

LADY ANNE:
Dost grant me, hedgehog? then, God grant me too
Thou mayst be damned for that wicked deed!
O, he was gentle, mild, and virtuous!

GLOUCESTER:
The fitter for the King of heaven, that hath him.

LADY ANNE:
He is in heaven, where thou shalt never come.

GLOUCESTER:
Let him thank me, or I am sure yet;
But send me France and my true Jove's hands.

WARWICK:
And I choose Clarence only for protector.

KING HENRY VI:
Warwick and Clarence give me both your hands:
Now join your hands together, and your father's death
And hardening of my brother's son
It reep her to crave the hire, here my mother
```
