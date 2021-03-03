from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Field
from django import forms

LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('de', 'German'),
        ('hu', 'Hungarian'),
        ('fr', 'French'),
        ('pl', 'Poland'),
        ('es', 'Spanish'),
    ]


class SearchIndividualForm(forms.Form):
    first_name = forms.CharField(label='First name', max_length=100)
    last_name = forms.CharField(label='Last name', max_length=100)
    # Multiple choice?
    language = forms.ChoiceField(
        widget=forms.Select(),
        choices=LANGUAGE_CHOICES,)
    region = forms.ChoiceField(
        widget=forms.Select(),
        choices=LANGUAGE_CHOICES,
    )

    # accuracy = forms.....
    helper = FormHelper()
    helper.layout = Layout(
        Field('first_name', css_class='input-xlarge'),
        Field('last_name', css_class='input-xlarge'),
        'language',
        'region',
        FormActions(
            Submit('submit', 'Submit', css_class="btn-primary"),
        )
    )


class SearchOrgForm(forms.Form):
    business_name = forms.CharField(label='Business name', max_length=200)
    legal_type = forms.CharField(label='Ltd., Inc., Bt., Sárl, etc.', max_length=100)
    # Multiple choice?
    language = forms.ChoiceField(
        widget=forms.Select(),
        choices=LANGUAGE_CHOICES,)
    region = forms.ChoiceField(
        widget=forms.Select(),
        choices=LANGUAGE_CHOICES,
    )

    # accuracy = forms.....
    # Uni-form
    helper = FormHelper()
    helper.layout = Layout(
        Field('business_name', css_class='input-xlarge'),
        Field('legal_type', css_class='input-xlarge'),
        'language',
        'region',
        FormActions(
            Submit('submit', 'Submit', css_class="btn-primary"),
        )
    )
