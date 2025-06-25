<script>
document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const area = parseFloat(formData.get('area'));
    const rooms_count = parseInt(formData.get('rooms_count'));
    const floors_count = parseInt(formData.get('floors_count'));
    const floor = parseInt(formData.get('floor'));

    const credentials = btoa("admin:1234");

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Authorization': `Basic ${credentials}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                area: area,
                rooms_count: rooms_count,
                floors_count: floors_count,
                floor: floor
            })
        });

        const data = await res.json();

        if (res.ok) {
            document.getElementById('response').textContent = `Predicted Price: ${data.price} ₽`;
        } else {
            document.getElementById('response').textContent = `Ошибка: ${data.error || "Неверный ввод"}`;
        }
    } catch (err) {
        document.getElementById('response').textContent = `Сетевая ошибка: ${err.message}`;
    }
});
</script>
