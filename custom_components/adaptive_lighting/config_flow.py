"""Config flow for Adaptive Lighting integration."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_NAME, MAJOR_VERSION, MINOR_VERSION
from homeassistant.core import callback
from homeassistant.helpers import selector

from . import const
from .const import DOMAIN, EXTRA_VALIDATION, NONE_STR, VALIDATION_TUPLES
from .switch import _supported_features, validate

_LOGGER = logging.getLogger(__name__)

DEFAULTS_BY_KEY = {key: default for key, default, _ in VALIDATION_TUPLES}
VALIDATION_BY_KEY = {key: validation for key, _, validation in VALIDATION_TUPLES}

BOOL_KEYS = {
    const.CONF_PREFER_RGB_COLOR,
    const.CONF_ADAPT_UNTIL_SLEEP,
    const.CONF_TAKE_OVER_CONTROL,
    const.CONF_DETECT_NON_HA_CHANGES,
    const.CONF_ONLY_ONCE,
    const.CONF_ADAPT_ONLY_ON_BARE_TURN_ON,
    const.CONF_SEPARATE_TURN_ON_COMMANDS,
    const.CONF_SKIP_REDUNDANT_COMMANDS,
    const.CONF_INTERCEPT,
    const.CONF_MULTI_LIGHT_INTERCEPT,
    const.CONF_INCLUDE_CONFIG_IN_ATTRIBUTES,
}

SELECTOR_OVERRIDES: dict[str, selector.Selector] = {
    key: selector.BooleanSelector() for key in BOOL_KEYS
}

SELECTOR_OVERRIDES.update(
    {
        const.CONF_INTERVAL: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                step=1,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_TRANSITION: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                step=1,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_INITIAL_TRANSITION: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                step=1,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_SLEEP_TRANSITION: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                step=1,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_MIN_BRIGHTNESS: selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=100,
                step=1,
                mode=selector.NumberSelectorMode.SLIDER,
                unit_of_measurement="%",
            ),
        ),
        const.CONF_MAX_BRIGHTNESS: selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=100,
                step=1,
                mode=selector.NumberSelectorMode.SLIDER,
                unit_of_measurement="%",
            ),
        ),
        const.CONF_SLEEP_BRIGHTNESS: selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1,
                max=100,
                step=1,
                mode=selector.NumberSelectorMode.SLIDER,
                unit_of_measurement="%",
            ),
        ),
        const.CONF_MIN_COLOR_TEMP: selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1000,
                max=10000,
                step=50,
                mode=selector.NumberSelectorMode.SLIDER,
                unit_of_measurement="K",
            ),
        ),
        const.CONF_MAX_COLOR_TEMP: selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1000,
                max=10000,
                step=50,
                mode=selector.NumberSelectorMode.SLIDER,
                unit_of_measurement="K",
            ),
        ),
        const.CONF_SLEEP_COLOR_TEMP: selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=1000,
                max=10000,
                step=50,
                mode=selector.NumberSelectorMode.SLIDER,
                unit_of_measurement="K",
            ),
        ),
        const.CONF_BRIGHTNESS_MODE_TIME_LIGHT: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                step=60,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_BRIGHTNESS_MODE_TIME_DARK: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                step=60,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_SUNRISE_OFFSET: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                min=-43200,
                max=43200,
                step=60,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_SUNSET_OFFSET: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                min=-43200,
                max=43200,
                step=60,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_AUTORESET_CONTROL: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                min=0,
                max=31536000,
                step=60,
                unit_of_measurement="s",
            ),
        ),
        const.CONF_SEND_SPLIT_DELAY: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                min=0,
                max=10000,
                step=10,
                unit_of_measurement="ms",
            ),
        ),
        const.CONF_ADAPT_DELAY: selector.NumberSelector(
            selector.NumberSelectorConfig(
                mode=selector.NumberSelectorMode.BOX,
                min=0,
                max=60,
                step=0.1,
                unit_of_measurement="s",
            ),
        ),
    },
)

STEP_FIELDS: dict[str, list[str]] = {
    "init": [
        const.CONF_LIGHTS,
        const.CONF_INTERVAL,
        const.CONF_TRANSITION,
        const.CONF_INITIAL_TRANSITION,
    ],
    "daytime": [
        const.CONF_MIN_BRIGHTNESS,
        const.CONF_MAX_BRIGHTNESS,
        const.CONF_MIN_COLOR_TEMP,
        const.CONF_MAX_COLOR_TEMP,
        const.CONF_PREFER_RGB_COLOR,
        const.CONF_BRIGHTNESS_MODE,
        const.CONF_BRIGHTNESS_MODE_TIME_LIGHT,
        const.CONF_BRIGHTNESS_MODE_TIME_DARK,
    ],
    "sleep": [
        const.CONF_ADAPT_UNTIL_SLEEP,
        const.CONF_SLEEP_BRIGHTNESS,
        const.CONF_SLEEP_RGB_OR_COLOR_TEMP,
        const.CONF_SLEEP_COLOR_TEMP,
        const.CONF_SLEEP_RGB_COLOR,
        const.CONF_SLEEP_TRANSITION,
    ],
    "sun": [
        const.CONF_SUNRISE_TIME,
        const.CONF_MIN_SUNRISE_TIME,
        const.CONF_MAX_SUNRISE_TIME,
        const.CONF_SUNRISE_OFFSET,
        const.CONF_SUNSET_TIME,
        const.CONF_MIN_SUNSET_TIME,
        const.CONF_MAX_SUNSET_TIME,
        const.CONF_SUNSET_OFFSET,
    ],
    "automation": [
        const.CONF_TAKE_OVER_CONTROL,
        const.CONF_DETECT_NON_HA_CHANGES,
        const.CONF_ONLY_ONCE,
        const.CONF_ADAPT_ONLY_ON_BARE_TURN_ON,
        const.CONF_SEPARATE_TURN_ON_COMMANDS,
        const.CONF_SEND_SPLIT_DELAY,
        const.CONF_ADAPT_DELAY,
        const.CONF_AUTORESET_CONTROL,
        const.CONF_SKIP_REDUNDANT_COMMANDS,
        const.CONF_INTERCEPT,
        const.CONF_MULTI_LIGHT_INTERCEPT,
        const.CONF_INCLUDE_CONFIG_IN_ATTRIBUTES,
    ],
}


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Adaptive Lighting."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            await self.async_set_unique_id(user_input[CONF_NAME])
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({vol.Required(CONF_NAME): str}),
            errors=errors,
        )

    async def async_step_import(self, user_input=None):
        """Handle configuration by YAML file."""
        await self.async_set_unique_id(user_input[CONF_NAME])
        # Keep a list of switches that are configured via YAML
        data = self.hass.data.setdefault(DOMAIN, {})
        data.setdefault("__yaml__", set()).add(self.unique_id)

        for entry in self._async_current_entries():
            if entry.unique_id == self.unique_id:
                self.hass.config_entries.async_update_entry(entry, data=user_input)
                self._abort_if_unique_id_configured()

        return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        if (MAJOR_VERSION, MINOR_VERSION) >= (2024, 12):
            # https://github.com/home-assistant/core/pull/129651
            return OptionsFlowHandler()
        return OptionsFlowHandler(config_entry)


def validate_options(user_input, errors):
    """Validate the options in the OptionsFlow.

    This is an extra validation step because the validators
    in `EXTRA_VALIDATION` cannot be serialized to json.
    """
    for key, (_validate, _) in EXTRA_VALIDATION.items():
        # these are unserializable validators
        value = user_input.get(key)
        try:
            if value is not None and value != NONE_STR:
                _validate(value)
        except vol.Invalid:
            _LOGGER.exception("Configuration option %s=%s is incorrect", key, value)
            errors["base"] = "option_error"


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle a option flow for Adaptive Lighting."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize options flow."""
        if (MAJOR_VERSION, MINOR_VERSION) >= (2024, 12):
            super().__init__(*args, **kwargs)
            # https://github.com/home-assistant/core/pull/129651
        else:
            self.config_entry = args[0]
        self._options: dict[str, Any] | None = None
        self._serialized_defaults: dict[str, Any] | None = None
        self._supported_lights: list[str] | None = None

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Configure basic settings and the light list."""
        if self.config_entry.source == config_entries.SOURCE_IMPORT:
            return self.async_show_form(step_id="init", data_schema=None)
        return await self._async_process_step(
            step_id="init",
            field_keys=STEP_FIELDS["init"],
            next_step=self.async_step_daytime,
            user_input=user_input,
        )

    async def async_step_daytime(self, user_input: dict[str, Any] | None = None):
        """Configure daytime brightness and color behaviour."""
        return await self._async_process_step(
            step_id="daytime",
            field_keys=STEP_FIELDS["daytime"],
            next_step=self.async_step_sleep,
            user_input=user_input,
        )

    async def async_step_sleep(self, user_input: dict[str, Any] | None = None):
        """Configure sleep mode behaviour."""
        return await self._async_process_step(
            step_id="sleep",
            field_keys=STEP_FIELDS["sleep"],
            next_step=self.async_step_sun,
            user_input=user_input,
        )

    async def async_step_sun(self, user_input: dict[str, Any] | None = None):
        """Configure virtual sun schedule."""
        return await self._async_process_step(
            step_id="sun",
            field_keys=STEP_FIELDS["sun"],
            next_step=self.async_step_automation,
            user_input=user_input,
        )

    async def async_step_automation(self, user_input: dict[str, Any] | None = None):
        """Configure automation safeguards and advanced options."""
        return await self._async_process_step(
            step_id="automation",
            field_keys=STEP_FIELDS["automation"],
            next_step=None,
            user_input=user_input,
        )

    async def _async_process_step(
        self,
        *,
        step_id: str,
        field_keys: list[str],
        next_step: Callable[[], Awaitable[config_entries.ConfigFlowResult]] | None,
        user_input: dict[str, Any] | None,
    ) -> config_entries.ConfigFlowResult:
        """Render a step, validate input, and advance when successful."""
        self._ensure_state()

        errors: dict[str, str] = {}
        if user_input is not None:
            normalized = self._normalize_user_input(field_keys, user_input)
            errors = self._validate_step(step_id, normalized)
            if not errors:
                assert self._options is not None
                self._options.update(normalized)
                if next_step is None:
                    return await self._finish(step_id)
                return await next_step()

        schema = self._build_schema(field_keys)
        return self.async_show_form(step_id=step_id, data_schema=schema, errors=errors)

    def _ensure_state(self) -> None:
        """Initialize cached state used across steps."""
        if self._options is not None:
            return

        validated = validate(self.config_entry)
        self._options = dict(self.config_entry.options)
        self._serialized_defaults = {
            key: self._serialize_value_for_form(key, value)
            for key, value in validated.items()
        }

        configured_lights = validated.get(const.CONF_LIGHTS, []) or []
        supported_lights = [
            light
            for light in self.hass.states.async_entity_ids("light")
            if _supported_features(self.hass, light)
        ]
        self._supported_lights = sorted({*supported_lights, *configured_lights})

    def _serialize_value_for_form(self, key: str, value: Any) -> Any:
        """Convert validated values to form-friendly representations."""
        if value is None:
            if DEFAULTS_BY_KEY.get(key) == NONE_STR:
                return NONE_STR
            return None

        if key in EXTRA_VALIDATION:
            _, coerce = EXTRA_VALIDATION[key]
            if coerce is not None:
                if isinstance(value, dt.timedelta):
                    return coerce(value)
                if isinstance(value, dt.time):
                    return coerce(value)

        return value

    def _normalize_user_input(
        self,
        field_keys: list[str],
        user_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize values coming back from the UI."""
        normalized: dict[str, Any] = {}
        for key in field_keys:
            if key not in user_input:
                continue
            value = user_input[key]

            if isinstance(value, str) and value.strip().lower() == "none" and (
                DEFAULTS_BY_KEY.get(key) == NONE_STR
            ):
                value = NONE_STR

            if isinstance(value, str) and value == "" and DEFAULTS_BY_KEY.get(key) == NONE_STR:
                value = NONE_STR

            default_value = DEFAULTS_BY_KEY.get(key)
            if isinstance(value, float) and type(default_value) is int:
                value = int(round(value))

            if key == const.CONF_LIGHTS and value is None:
                value = []

            normalized[key] = value

        return normalized

    def _validate_step(
        self,
        step_id: str,
        user_input: dict[str, Any],
    ) -> dict[str, str]:
        """Perform step-specific validation."""
        errors: dict[str, str] = {}

        if step_id == "init":
            lights = user_input.get(const.CONF_LIGHTS)
            if lights is None and self._options is not None:
                lights = self._options.get(const.CONF_LIGHTS)
            if lights:
                assert self._supported_lights is not None
                missing = [light for light in lights if light not in self._supported_lights]
                if missing:
                    errors[const.CONF_LIGHTS] = "entity_missing"
                    title = self.config_entry.title or self.config_entry.data.get(CONF_NAME)
                    for light in missing:
                        _LOGGER.error(
                            "%s: light entity %s is configured, but was not found",
                            title,
                            light,
                        )
                    self._supported_lights = sorted({*self._supported_lights, *missing})

        return errors

    def _build_schema(self, field_keys: list[str]) -> vol.Schema:
        """Build a schema for the current step."""
        assert self._options is not None
        assert self._serialized_defaults is not None

        schema_dict: dict[Any, Any] = {}
        for key in field_keys:
            current_value = self._options.get(key, self._serialized_defaults.get(key))
            if current_value is None and DEFAULTS_BY_KEY.get(key) == NONE_STR:
                current_value = NONE_STR

            if key == const.CONF_LIGHTS:
                assert self._supported_lights is not None
                default_value = current_value or []
                schema_dict[vol.Optional(key, default=default_value)] = selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="light",
                        multiple=True,
                        include_entities=self._supported_lights,
                    ),
                )
                continue

            field_component = SELECTOR_OVERRIDES.get(key, VALIDATION_BY_KEY[key])
            schema_dict[vol.Optional(key, default=current_value)] = field_component

        return vol.Schema(schema_dict)

    async def _finish(self, final_step_id: str) -> config_entries.ConfigFlowResult:
        """Validate all settings and persist the options entry."""
        assert self._options is not None

        errors: dict[str, str] = {}
        validate_options(self._options, errors)
        if errors:
            schema = self._build_schema(STEP_FIELDS[final_step_id])
            return self.async_show_form(step_id=final_step_id, data_schema=schema, errors=errors)

        return self.async_create_entry(title="", data=dict(self._options))
