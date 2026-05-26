DO $$
BEGIN
  IF to_regclass('public.stock_prices') IS NOT NULL
     AND to_regclass('public.raw_stock_prices') IS NULL THEN
    ALTER TABLE stock_prices RENAME TO raw_stock_prices;
  END IF;
END $$;
