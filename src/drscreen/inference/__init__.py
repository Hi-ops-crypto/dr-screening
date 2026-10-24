"""Screening of a single photograph: validation, quality gate, grading, referral.

Works from checkpoints alone and never imports training, so the deployed app
carries no optimiser, augmentation or dataset code.
"""
