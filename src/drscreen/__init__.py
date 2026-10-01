"""Diabetic retinopathy screening prototype.

Subpackages are imported explicitly by their consumers; this module
deliberately re-exports nothing so that importing a lightweight part of the
package (for example the risk calculator) never pulls in torch.
"""
