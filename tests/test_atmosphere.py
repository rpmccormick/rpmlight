from rpmlight.atmosphere import (
    decode_cast_message_body,
    delay_s_to_slider_pct,
    encode_cast_message,
    is_allowed_personal_album_url,
    parse_google_photos_album_html,
    slider_pct_to_delay_s,
)


def test_atmosphere_slider_curve_and_roundtrip() -> None:
    assert slider_pct_to_delay_s(1) == 300.0
    assert slider_pct_to_delay_s(50) == 45.0
    assert slider_pct_to_delay_s(100) == 5.0
    assert delay_s_to_slider_pct(300.0) == 1
    assert delay_s_to_slider_pct(45.0) == 50
    assert delay_s_to_slider_pct(5.0) == 100


def test_atmosphere_cast_framing_and_personal_album_validation() -> None:
    pkt = encode_cast_message(
        "sender-0", "receiver-0", "urn:x-cast:test", '{"ok":true}'
    )
    assert decode_cast_message_body(pkt[4:]) == (
        "sender-0",
        "receiver-0",
        "urn:x-cast:test",
        '{"ok":true}',
    )
    assert is_allowed_personal_album_url("https://photos.app.goo.gl/AbCdEf123") is True
    assert is_allowed_personal_album_url("http://127.0.0.1/bad") is False

    html = (
        'AF_initDataCallback({data:[["1",'
        '["https://lh3.googleusercontent.com/pw/AP1GczSampleLandscape01",1920,1080]]]})'
    )
    assert parse_google_photos_album_html(html) == [
        "https://lh3.googleusercontent.com/pw/AP1GczSampleLandscape01=w1920-h1080"
    ]
