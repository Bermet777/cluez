# Generated by Django 3.0.12 on 2021-02-23 15:17

import django.contrib.postgres.fields.hstore
from django.db import migrations, models
from django.contrib.postgres.operations import HStoreExtension


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0001_initial'),
    ]

    operations = [
        HStoreExtension(),
        migrations.AddField(
            model_name='search',
            name='result',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='searchresult',
            name='data',
            field=django.contrib.postgres.fields.hstore.HStoreField(default=None),
            preserve_default=False,
        ),
    ]
